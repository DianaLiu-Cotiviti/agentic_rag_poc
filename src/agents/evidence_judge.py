"""
Evidence Judge Agent

Responsible for evaluating the quality of retrieved evidence, determining:
1. Whether evidence sufficiently answers the question (is_sufficient)
2. Evidence coverage (coverage_score)
3. Evidence relevance and accuracy (specificity_score)
4. Whether contradictory information exists (has_contradiction)
5. Missing aspects (missing_aspects)
"""

import logging
import sys
logger = logging.getLogger("agenticrag.evidence_judge")
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

from typing import List
from pydantic import BaseModel, Field
from .base import BaseAgent
from ..state import AgenticRAGState, RetrievalResult
from ..prompts.evidence_judge_prompts import (
    EVIDENCE_JUDGE_SYSTEM_MESSAGE,
    build_evidence_judgment_prompt
)
from ..utils.chunk_formatting import format_chunks_for_judge
from ..utils.save_workflow_outputs import save_top10_chunks


class EvidenceJudgment(BaseModel):
    """
    Evidence Judge Evaluation Result
    
    Judgment Criteria:
    - is_sufficient: Whether evidence is sufficient to answer the question (considering quantity, quality, coverage)
    - coverage_score: Evidence coverage of different aspects of the question (0.0-1.0)
    - specificity_score: Evidence specificity and accuracy (0.0-1.0)
    - has_contradiction: Whether contradictory information exists in retrieval results
    - missing_aspects: Aspects of the question not covered (to guide retry)
    - reasoning: Evaluation reasoning process (explaining why sufficient/insufficient)
    """
    is_sufficient: bool = Field(
        description="Whether the evidence is sufficient to answer the question"
    )
    coverage_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How well the evidence covers different aspects of the question (0.0-1.0)"
    )
    specificity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How specific and accurate the evidence is (0.0-1.0)"
    )
    has_contradiction: bool = Field(
        description="Whether there are contradictory statements in the evidence"
    )
    missing_aspects: List[str] = Field(
        default_factory=list,
        description="Aspects of the question not covered by current evidence"
    )
    reasoning: str = Field(
        description="Detailed reasoning for the sufficiency judgment"
    )


class EvidenceJudgeAgent(BaseAgent):
    """
    Evidence Judge Agent - Evaluate retrieval evidence quality
    
    Core Responsibilities:
    1. Determine if evidence is sufficient (is_sufficient)
       - Consider question type (simple CPT lookup vs complex billing rules)
       - Consider evidence quantity and quality
       - Consider coverage
    
    2. Evaluate evidence quality metrics:
       - coverage_score: Coverage of multiple aspects of the question (CPT code definition, modifier, bundling, etc.)
       - specificity_score: Evidence accuracy and relevance
    
    3. Identify issues:
       - has_contradiction: Detect contradictory information
       - missing_aspects: Identify missing aspects
    
    4. Guide next actions:
       - If insufficient, missing_aspects guides query refinement
       - If sufficient, high-quality chunks used for answer generation
    """
    
    def __init__(self, config, client=None):
        """
        Args:
            config: Configuration object with Azure OpenAI settings
            client: Azure OpenAI client (optional, will use config.client if not provided)
        """
        self.config = config
        self._client = client if client is not None else getattr(config, 'client', None)
        
        # Initialize retrieval tools for cross-encoder reranking
        from ..tools.retrieval_tools import RetrievalTools
        self.retrieval_tools = RetrievalTools(config)
    
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
        Evaluate retrieval evidence quality
        
        Args:
            state: Contains question, question_type, retrieved_chunks, cpt_descriptions
            
        Returns:
            dict: Contains evidence_assessment
        """
        question = state["question"]
        question_type = state.get("question_type", "general")
        chunks = state.get("retrieved_chunks", [])
        retrieval_metadata = state.get("retrieval_metadata", {})
        cpt_descriptions = state.get("cpt_descriptions", {})  # Get CPT descriptions from state
        
        # If no chunks retrieved - clearly insufficient
        if not chunks:
            return {
                "evidence_assessment": {
                    "is_sufficient": False,
                    "coverage_score": 0.0,
                    "specificity_score": 0.0,
                    "has_contradiction": False,
                    "missing_aspects": ["No chunks retrieved - all aspects missing"],
                    "reasoning": "No relevant chunks were retrieved. Need to refine query or adjust retrieval strategy."
                }
            }
        
        # Apply cross-encoder reranking if enabled
        reranked_chunks = chunks  # Keep original for comparison
        if self.config.use_cross_encoder_rerank and len(chunks) > self.config.cross_encoder_top_k:
            logger.info(f"\n🔄 Layer 3 Reranking: Cross-Encoder (Question-aware)")
            logger.info(f"   Purpose: Refine {len(chunks)} chunks to top {self.config.cross_encoder_top_k} based on original question")
            logger.info(f"   Before: {len(chunks)} chunks (from Layer 1-2 fusion)")
            
            # Call cross-encoder reranking tool
            reranked_chunks = self.retrieval_tools.cross_encoder_rerank(
                query=question,
                chunks=chunks,
                top_k=self.config.cross_encoder_top_k
            )
            
            logger.info(f"   After: {len(reranked_chunks)} chunks (optimized for Evidence Judge)")
            
            # Update metadata
            retrieval_metadata["cross_encoder_reranked"] = True
            retrieval_metadata["cross_encoder_model"] = self.config.cross_encoder_model
            retrieval_metadata["chunks_before_layer3"] = len(chunks)
            retrieval_metadata["chunks_after_layer3"] = len(reranked_chunks)
            
            # Save top 10 chunks as basis for LLM response
            mode = retrieval_metadata.get('mode', 'unknown')
            save_path = save_top10_chunks(
                top10_chunks=reranked_chunks,
                question=state.get('question', ''),
                output_dir=self.config.retrieval_output_dir,
                metadata={
                    'mode': mode,
                    'original_chunks_count': len(chunks),
                    'reranked_to_top': len(reranked_chunks),
                    'layer': 'layer3_cross_encoder'
                }
            )
            logger.info(f"   💾 Top 10 chunks saved to: {save_path}")
        else:
            # Cross-encoder disabled or not enough chunks - use score-based top-K
            if not self.config.use_cross_encoder_rerank:
                logger.info(f"\n⏭️  Layer 3 Reranking: Skipped (disabled in config)")
            else:
                logger.info(f"\n⏭️  Layer 3 Reranking: Skipped (only {len(chunks)} chunks, threshold is {self.config.cross_encoder_top_k})")
            
            # Still limit to top-K based on existing scores (from Layer 1-2)
            if len(chunks) > self.config.cross_encoder_top_k:
                reranked_chunks = chunks[:self.config.cross_encoder_top_k]
                logger.info(f"   📊 Using top {self.config.cross_encoder_top_k} chunks based on Layer 1-2 scores")
                retrieval_metadata["cross_encoder_reranked"] = False
                retrieval_metadata["truncated_to_top_k"] = True
            else:
                logger.info(f"   📊 Using all {len(chunks)} chunks (no truncation needed)")
                retrieval_metadata["cross_encoder_reranked"] = False
            
            # Save top 10 chunks as basis for LLM response (score-based version)
            mode = retrieval_metadata.get('mode', 'unknown')
            save_path = save_top10_chunks(
                top10_chunks=reranked_chunks,
                question=state.get('question', ''),
                output_dir=self.config.retrieval_output_dir,
                metadata={
                    'mode': mode,
                    'original_chunks_count': len(chunks),
                    'reranked_to_top': len(reranked_chunks),
                    'layer': 'layer1_layer2_score_based'
                }
            )
            logger.info(f"   💾 Top {len(reranked_chunks)} chunks saved to: {save_path}")

        
        # Use reranked chunks for evaluation
        chunks_to_judge = reranked_chunks
        
        # Format chunks for prompt
        chunks_text = format_chunks_for_judge(chunks_to_judge, cpt_descriptions=cpt_descriptions)
        
        # Build prompt for LLM evaluation
        # Note: Only use original question and retrieved chunks for evaluation
        # Sub-queries are not needed (they are just retrieval means, not evaluation targets)
        prompt = build_evidence_judgment_prompt(
            question=question,
            question_type=question_type,
            chunks_text=chunks_text,
            retrieval_mode=retrieval_metadata.get("mode", "unknown"),
            strategies_used=", ".join(retrieval_metadata.get("strategies_used", [])),
            total_chunks=len(chunks_to_judge)
        )
        
        # Call LLM for structured evaluation
        response = self.client.beta.chat.completions.parse(
            model=self.config.azure_deployment_name,
            messages=[
                {"role": "system", "content": EVIDENCE_JUDGE_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            response_format=EvidenceJudgment,
            temperature=self.config.agent_temperature,
            max_tokens=2500  # Increased for detailed evidence assessment
        )
        
        judgment = response.choices[0].message.parsed
        
        # Return updated state with reranked chunks
        return {
            "evidence_assessment": {
                "is_sufficient": judgment.is_sufficient,
                "coverage_score": judgment.coverage_score,
                "specificity_score": judgment.specificity_score,
                "has_contradiction": judgment.has_contradiction,
                "missing_aspects": judgment.missing_aspects,
                "reasoning": judgment.reasoning
            },
            "retrieved_chunks": reranked_chunks,  # Update state with top-10 reranked chunks
            "retrieval_metadata": retrieval_metadata  # Update metadata with Layer 3 info
        }


