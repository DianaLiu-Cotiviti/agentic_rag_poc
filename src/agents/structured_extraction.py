"""
Structured Extraction Agent - Structured Extractor
Responsible for extracting structured answers from evidence
"""

from pydantic import BaseModel, Field
from .base import BaseAgent
from ..state import AgenticRAGState


class StructuredAnswer(BaseModel):
    """Structured answer"""
    answer: str = Field(description="Complete answer")
    confidence_score: float = Field(ge=0, le=1, description="Confidence level")
    citations: list[dict] = Field(description="Citation sources")
    reasoning: str = Field(description="Reasoning process")


class StructuredExtractionAgent(BaseAgent):
    """
    Structured Extraction Agent
    
    Responsibilities:
    1. Generate complete answer based on evidence
    2. Provide citation sources
    3. Calculate confidence level
    4. Explain reasoning process
    """
    
    def process(self, state: AgenticRAGState) -> dict:
        """
        Extract structured answer
        
        Args:
            state: State containing question and retrieved_chunks
            
        Returns:
            dict: Contains structured_answer
        """
        question = state["question"]
        chunks = state.get("retrieved_chunks", [])
        
        # Build prompt
        prompt = self._build_prompt(question, chunks)
        
        # Call LLM to extract answer
        response = self.client.beta.chat.completions.parse(
            model=self.config.azure_deployment_name,
            messages=[
                {"role": "system", "content": "You are a medical coding expert providing accurate answers based on evidence."},
                {"role": "user", "content": prompt}
            ],
            response_format=StructuredAnswer,
            temperature=self.config.agent_temperature
        )
        
        answer = response.choices[0].message.parsed
        
        return {
            "structured_answer": {
                "answer": answer.answer,
                "confidence_score": answer.confidence_score,
                "citations": answer.citations,
                "reasoning": answer.reasoning
            }
        }
    
    def _build_prompt(self, question: str, chunks: list) -> str:
        """Build Structured Extraction prompt"""
        evidence = "\n\n".join([
            f"[Document {i+1}] {chunk.get('text', '')}"
            for i, chunk in enumerate(chunks)
        ])
        
        return f"""Answer the question based on the following evidence:

Question: {question}

Evidence:
{evidence}

Please provide:
1. Complete and accurate answer
2. Confidence score (0-1)
3. Cited document sources
4. Your reasoning process
"""
