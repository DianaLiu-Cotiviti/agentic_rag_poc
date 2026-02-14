"""
Query Refiner Agent

Generates refined queries based on Evidence Judge assessment
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from .base import BaseAgent
from src.state import AgenticRAGState, RetrievalResult, EvidenceAssessment
from src.prompts.query_refiner_prompts import (
    QUERY_REFINER_SYSTEM_MESSAGE,
    build_query_refiner_prompt
)
import logging
import sys
logger = logging.getLogger("agenticrag.query_refiner")
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class RefinedQuery(BaseModel):
    """Refined query targeting specific missing aspect"""
    query: str = Field(description="Refined query text", max_length=300)
    target_aspect: str = Field(description="Missing aspect this query addresses", max_length=200)
    retrieval_hint: Optional[str] = Field(
        default=None,
        max_length=150,  # Limit hint length to prevent prompt bloat
        description="Hint for retrieval strategy (e.g., 'focus on modifiers')"
    )


class QueryRefinementResult(BaseModel):
    """Result of query refinement"""
    refined_queries: List[RefinedQuery] = Field(
        description="List of refined queries, one per missing aspect"
    )
    reasoning: str = Field(description="Explanation of refinement strategy")


class QueryRefinerAgent(BaseAgent):
    """
    Query Refiner Agent
    
    Analyzes insufficient evidence and generates targeted refined queries
    to address missing aspects identified by Evidence Judge.
    
    Responsibilities:
    - Analyze missing_aspects from Evidence Judge
    - Generate specific queries targeting each gap
    - Avoid redundancy with previous queries
    - Select top chunks to preserve for merging
    """
    
    def process(self, state: AgenticRAGState) -> AgenticRAGState:
        """
        Generate refined queries based on insufficient evidence
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with refined queries and keep_chunks
        """
        logger.info("\n" + "="*80)
        logger.info("🔄 QUERY REFINER - Generating Refined Queries")
        logger.info("="*80)
        
        # Extract context
        question = state["question"]
        evidence_assessment = state.get("evidence_assessment", {})
        missing_aspects = evidence_assessment.get("missing_aspects", [])  # ← Get from evidence_assessment
        
        # Get previous queries
        previous_queries = []
        if state.get("query_candidates"):
            # query_candidates can be QueryCandidate objects or dicts
            previous_queries.extend([
                qc.get("query") if isinstance(qc, dict) else qc.query
                for qc in state["query_candidates"]
            ])
        if state.get("refined_queries"):
            # refined_queries are always dicts
            previous_queries.extend([rq["query"] for rq in state["refined_queries"]])
            
        # Get current retry count
        retry_count = state.get("retry_count", 0)
        
        logger.info(f"\n📊 Context:")
        logger.info(f"   - Missing Aspects: {len(missing_aspects)}")
        logger.info(f"   - Previous Queries: {len(previous_queries)}")
        logger.info(f"   - Current Retry: {retry_count}")
        
        # Build prompt
        prompt = build_query_refiner_prompt(
            original_question=question,
            missing_aspects=missing_aspects,
            previous_queries=previous_queries,
            evidence_assessment=evidence_assessment
        )
        
        # Call LLM with structured output (consistent with Query Planner)
        logger.info("\n🤖 Calling LLM for query refinement (Structured Output)...")
        
        response = self.client.beta.chat.completions.parse(
            model=self.config.azure_deployment_name,
            messages=[
                {"role": "system", "content": QUERY_REFINER_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            response_format=QueryRefinementResult,  # Native structured output
            temperature=0.3,  # Lower for more focused refinement
            max_tokens=2000  # Increased for multiple query variants
        )
        
        # Parse result (automatic parsing, consistent with Query Planner)
        result = response.choices[0].message.parsed
        
        # Display results
        logger.info(f"\n✅ Generated {len(result.refined_queries)} refined queries:")
        for i, rq in enumerate(result.refined_queries, 1):
            logger.info(f"\n   Query {i}:")
            logger.info(f"   📝 Query: {rq.query}")
            logger.info(f"   🎯 Target: {rq.target_aspect}")
            if rq.retrieval_hint:
                logger.info(f"   💡 Hint: {rq.retrieval_hint}")
        
        logger.info(f"\n💭 Reasoning:\n{result.reasoning}")
        
        # Select top chunks to keep (adaptive selection, not fixed top-3)
        retrieved_chunks = state.get("retrieved_chunks", [])
        keep_chunks = self._select_keep_chunks(retrieved_chunks, min_keep=1, max_keep=5)
        
        logger.info(f"\n📦 Selected {len(keep_chunks)} chunks to preserve:")
        for i, chunk in enumerate(keep_chunks, 1):
            chunk_id = chunk.chunk_id if hasattr(chunk, 'chunk_id') else f'chunk_{i}'
            chunk_score = chunk.score if hasattr(chunk, 'score') else 0
            logger.info(f"   {i}. {chunk_id} (score: {chunk_score:.4f})")
        
        # Store previous assessment for improvement validation
        previous_assessment = state.get("evidence_assessment")
        
        # Increment retry count
        new_retry_count = retry_count + 1
        
        logger.info(f"\n🔄 Retry count incremented: {retry_count} → {new_retry_count}")
        
        # Update state
        return {
            **state,
            "refined_queries": [rq.model_dump() for rq in result.refined_queries],
            "keep_chunks": keep_chunks,  # Keep as RetrievalResult objects
            "previous_assessment": previous_assessment,
            "retry_count": new_retry_count
        }
    
    def _select_keep_chunks(
        self,
        retrieved_chunks: List[RetrievalResult],
        min_keep: int = 1,
        max_keep: int = 5
    ) -> List[RetrievalResult]:
        """
        Adaptive chunk selection - OpenAI/Anthropic standard method
        
        Three Core Strategies (fully consistent with OpenAI RAG):
        
        1️⃣ Score Threshold:
           - Keep chunks with score >= max_score × 0.85
           - Relative threshold adapts to different retrieval quality
        
        2️⃣ Score Gap Detection (cliff detection):
           - Detect adjacent chunk score difference > 0.10
           - Stop before cliff, preserve high-quality cluster
        
        3️⃣ Quality Floor:
           - Keep minimum 1, maximum 5 chunks
           - Ensures baseline, avoids excessive retention
        
        Real-world examples:
        - Top 5 all similar (0.95, 0.94, 0.93, 0.92, 0.91) → Keep all 5 ✅
        - Top 3 high, rest low (0.95, 0.94, 0.93, 0.60, 0.55) → Keep top 3 ✅
        - Same document dominates → Diversity enforcement limits to 2 ✅
        
        Args:
            retrieved_chunks: All retrieved chunks from previous round
            min_keep: Minimum chunks to keep (default: 1)
            max_keep: Maximum chunks to keep (default: 5)
            
        Returns:
            Adaptively selected chunks sorted by score
        """
        if not retrieved_chunks:
            return []
        
        # Sort by score descending
        # retrieved_chunks are RetrievalResult objects, not dicts
        sorted_chunks = sorted(
            retrieved_chunks,
            key=lambda x: x.score if hasattr(x, 'score') else 0,
            reverse=True
        )
        
        if len(sorted_chunks) == 0:
            return []
        
        # ========== OpenAI Three Strategies ==========
        
        # 1️⃣ Score Threshold: score >= max_score × 0.85
        max_score = sorted_chunks[0].score if hasattr(sorted_chunks[0], 'score') else 0
        score_threshold = max_score * 0.85
        
        logger.info(f"      [OpenAI Strategy] Score threshold: {score_threshold:.4f} (85% of max: {max_score:.4f})")
        
        # 2️⃣ Score Gap Detection + Threshold filtering
        selected = []
        for i, chunk in enumerate(sorted_chunks[:max_keep]):
            chunk_score = chunk.score if hasattr(chunk, 'score') else 0
            
            # Check score threshold (Strategy 1)
            if chunk_score < score_threshold:
                logger.info(f"      [Threshold Stop] Chunk #{i+1} score {chunk_score:.4f} < threshold {score_threshold:.4f}")
                break
            
            # Check score gap (Strategy 2) - if not first chunk
            if i > 0:
                prev_score = sorted_chunks[i-1].score if hasattr(sorted_chunks[i-1], 'score') else 0
                score_gap = prev_score - chunk_score
                
                # Large gap detected (>0.10) → stop here
                if score_gap > 0.10:
                    logger.info(f"      [Gap Stop] Score gap: {prev_score:.4f} → {chunk_score:.4f} (Δ={score_gap:.4f} > 0.10)")
                    break
            
            selected.append(chunk)
        
        # 3️⃣ Quality Floor: min_keep <= len(selected) <= max_keep
        if len(selected) < min_keep and len(sorted_chunks) >= min_keep:
            logger.info(f"      [Quality Floor] Enforcing min_keep={min_keep} (had {len(selected)})")
            selected = sorted_chunks[:min_keep]
        
        # Extra: Diversity enforcement (max 2 per document)
        selected = self._enforce_diversity_in_selection(selected, max_per_doc=2)
        
        logger.info(f"      [Result] {len(selected)} chunks kept (from {len(sorted_chunks)} total)")
        if selected:
            first_score = selected[0].score if hasattr(selected[0], 'score') else 0
            last_score = selected[-1].score if hasattr(selected[-1], 'score') else 0
            logger.info(f"      Score range: {first_score:.4f} - {last_score:.4f}")
        
        return selected
    
    def _enforce_diversity_in_selection(
        self,
        chunks: List[RetrievalResult],
        max_per_doc: int = 2
    ) -> List[RetrievalResult]:
        """
        Enforce diversity: limit chunks per document
        
        If top 5 chunks are all from the same document,
        only keep the best 2 from that document.
        
        Args:
            chunks: Chunks to filter
            max_per_doc: Max chunks per document (default: 2)
            
        Returns:
            Diversity-enforced chunks
        """
        if not chunks:
            return []
        
        doc_counts = {}
        filtered = []
        
        for chunk in chunks:
            # Access document_id from metadata if available
            doc_id = chunk.metadata.get("document_id", "unknown") if hasattr(chunk, 'metadata') and chunk.metadata else "unknown"
            current_count = doc_counts.get(doc_id, 0)
            
            if current_count < max_per_doc:
                filtered.append(chunk)
                doc_counts[doc_id] = current_count + 1
            # else: skip this chunk (too many from same doc)
        
        # Report diversity enforcement
        if len(filtered) < len(chunks):
            removed = len(chunks) - len(filtered)
            logger.info(f"      Diversity enforcement: Removed {removed} chunks (max {max_per_doc} per document)")
        
        return filtered

