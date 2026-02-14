""" Answer Generator Agent

Generates final answers based on top 10 chunks judged as sufficient by Evidence Judge.

Core Responsibilities:
1. Receive original question + top 10 high-quality chunks
2. Generate structured, evidence-supported answers
3. Cite specific chunk sources
4. Ensure answers are accurate, complete, and traceable

Design Principles:
- Answers must be based on provided chunks (no hallucination)
- Must cite specific chunks (traceability)
- If evidence is insufficient for certain aspects, state clearly (limitations)
- Code simplicity: prompts in prompts/, formatting in utils/
"""

from typing import Dict, Any
from pydantic import BaseModel, Field
from typing import Literal, List, Dict
from .base import BaseAgent
from ..state import AgenticRAGState
from ..prompts.answer_generator_prompts import (
    ANSWER_GENERATOR_SYSTEM_MESSAGE,
    build_answer_generation_prompt
)
from ..utils.chunk_formatting import (
    format_chunks_with_ids,
    format_cpt_descriptions,
    get_chunk_text_by_id
)


class Citation(BaseModel):
    """Single citation mapping citation number to chunk_id"""
    number: int = Field(description="Citation number used in answer, e.g., 1 for [1]")
    chunk_id: str = Field(description="Chunk ID being cited, e.g., 'chunk_000210'")


class CitedAnswer(BaseModel):
    """
    Output structure of Answer Generator
    
    Contains answer text and evidence citations (using numbered citation format [1] [2] [3])
    
    Note: citation_map is not in this model, it will be auto-generated in process()
    """
    answer: str = Field(
        description="Comprehensive answer with inline numbered citations [1] [2] [3]. MUST include citations after each claim."
    )
    key_points: List[str] = Field(
        default_factory=list,
        description="3-5 high-level summary bullet points (NOT sentence extraction, but executive summary of main takeaways). Each with citations [1] [2]."
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of citations mapping citation numbers to chunk IDs. E.g., [{number: 1, chunk_id: 'chunk_000210'}, {number: 2, chunk_id: 'chunk_000345'}]"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score based on evidence quality (0.0-1.0)"
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Any limitations or caveats in the answer"
    )


class AnswerGeneratorAgent(BaseAgent):
    """
    Answer Generator Agent - Generate Final Answer
    
    Workflow:
    1. Receive original question and top 10 chunks (verified as sufficient by Evidence Judge)
    2. Use prompt from prompts/answer_generator_prompts.py
    3. Return structured answer (containing citations, key_points, confidence)
    
    Token Optimization Strategy:
    - Only receive top 10 chunks verified as sufficient (via conditional edge)
    - Don't repeat evidence_assessment display (already done in Evidence Judge)
    - CPT descriptions formatted separately to avoid redundancy
    """
    
    def __init__(self, config, client=None):
        """
        Args:
            config: Configuration object with Azure OpenAI settings
            client: Azure OpenAI client (optional, will use config.client if not provided)
        """
        self.config = config
        self._client = client if client is not None else getattr(config, 'client', None)
    
    @property
    def client(self):
        """Lazy initialization of LLM client"""
        if self._client is None:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                api_key=self.config.azure_openai_api_key,
                api_version=self.config.azure_api_version,
                azure_endpoint=self.config.azure_openai_endpoint
            )
        return self._client
    
    def process(self, state: AgenticRAGState) -> dict:
        """
        Generate evidence-based answer
        
        Note: This method is only called when evidence is_sufficient=True (via conditional edge)
        Therefore no need to check evidence_assessment quality score again
        
        Args:
            state: Contains question, retrieved_chunks (top 10), cpt_descriptions
            
        Returns:
            dict: Contains final_answer
        """
        import logging
        logger = logging.getLogger("agenticrag.workflow_simple")
        logger.info("\nStart generating answer...")
        question = state["question"]
        chunks = state.get("retrieved_chunks", [])
        cpt_descriptions = state.get("cpt_descriptions", {})
        logger.info(f"Question: {question}")
        logger.info(f"Retrieved chunks: {len(chunks)}")

        # Check if chunks were retrieved
        if not chunks:
            logger.info("No chunks found, cannot generate answer.")
            return {
                "final_answer": {
                    "answer": "Unable to generate answer: No relevant evidence retrieved.",
                    "key_points": [],
                    "citations": [],
                    "confidence": 0.0,
                    "limitations": ["No relevant documents retrieved"]
                }
            }

        # Format chunks and CPT descriptions
        chunks_text = format_chunks_with_ids(chunks)
        cpt_desc_text = format_cpt_descriptions(cpt_descriptions)

        # Build prompt
        prompt = build_answer_generation_prompt(
            question=question,
            chunks_text=chunks_text,
            cpt_descriptions_text=cpt_desc_text
        )

        # Call LLM to generate answer
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.config.azure_deployment_name,
                messages=[
                    {"role": "system", "content": ANSWER_GENERATOR_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                ],
                response_format=CitedAnswer,
                temperature=self.config.agent_temperature,
                max_tokens=8000  # Maximum for comprehensive final answers
            )
            answer = response.choices[0].message.parsed
        except Exception as e:
            logger.info(f"LLM call failed: {e}")
            return {
                "final_answer": {
                    "answer": f"Answer generation failed: {e}",
                    "key_points": [],
                    "citations": [],
                    "confidence": 0.0,
                    "limitations": [str(e)]
                }
            }

        # Build citation_map
        citation_map = {}
        for citation in answer.citations:
            chunk_id = citation.chunk_id
            chunk_text = get_chunk_text_by_id(chunks, chunk_id)
            citation_map[citation.number] = {
                'chunk_id': chunk_id,
                'chunk_text': chunk_text
            }

        # Process limitations
        limitations = answer.limitations
        if limitations and all(isinstance(l, (int, float)) or (isinstance(l, str) and l.strip().isdigit()) for l in limitations):
            limitations = ["LLM did not provide specific limitations."]

        logger.info("Answer generation complete.")
        return {
            "final_answer": {
                "answer": answer.answer,
                "key_points": answer.key_points,
                "citation_map": citation_map,  # number -> {chunk_id, chunk_text}
                "confidence": answer.confidence,
                "limitations": limitations
            }
        }

