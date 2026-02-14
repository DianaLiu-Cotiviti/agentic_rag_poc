import logging
import sys
logger = logging.getLogger("agenticrag.retrieval_router_planning")
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


"""
Planning Retrieval Router - LLM generates plan, Agent executes

Characteristics:
- 1 LLM call (generate plan)
- Agent executes hardcoded plan
- Medium speed (~2 seconds)
- Medium cost ($0.01)
- LLM cannot adjust based on intermediate results (plan is fixed)
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from ..state import AgenticRAGState, RetrievalResult
from ..prompts.retrieval_router_prompts import (
    build_retrieval_router_prompt,
    RETRIEVAL_ROUTER_SYSTEM_MESSAGE
)


# Pydantic Schemas for Planning Mode
class PreFilteringConfig(BaseModel):
    """Pre-retrieval filtering configuration"""
    apply_range_routing: bool = Field(description="Whether to apply range routing pre-filter")
    range_filter_cpt_codes: List[str] = Field(default_factory=list, description="CPT codes for range filtering")
    range_filter_limit: int = Field(default=300, ge=100, le=1000, description="Max chunks from range routing")


class QueryStrategyMapping(BaseModel):
    """Maps a query candidate to its best retrieval strategy
    
    Each query selects the most suitable strategy based on characteristics:
    - hybrid: BM25+Semantic+RRF fusion (suitable for complex queries)
    - bm25: Pure keyword search (suitable for CPT codes, exact matching)
    - semantic: Pure semantic search (suitable for conceptual queries)
    - bm25_semantic: BM25 then Semantic, keeping both result sets (suitable for queries requiring dual verification)
    """
    query_index: int = Field(description="Index in query_candidates list")
    strategy: str = Field(
        description="Which retrieval strategy: hybrid, bm25, semantic, or bm25_semantic"
    )
    reasoning: str = Field(description="Why this strategy for this query")


class RetrievalParameters(BaseModel):
    """Retrieval parameters for each strategy"""
    bm25_top_k: int = Field(default=20, description="Top-k for BM25 search")
    semantic_top_k: int = Field(default=20, description="Top-k for semantic search")
    hybrid_top_k: int = Field(default=20, description="Top-k for hybrid search")
    hybrid_bm25_weight: float = Field(default=0.5, ge=0, le=1, description="BM25 weight in hybrid")
    hybrid_semantic_weight: float = Field(default=0.5, ge=0, le=1, description="Semantic weight in hybrid")


class FusionParameters(BaseModel):
    """Result fusion configuration (for multi-query fusion only)"""
    boost_range_results: float = Field(
        default=2.0,  # ← Increase default to 2.0
        ge=1.0,       # ← Minimum 1.0 (no score reduction)
        le=5.0,       # ← Increase max to 5.0, allowing stronger boost
        description="Boost factor for range-routed chunks (higher = prioritize pre-filtered results more)"
    )


class RetrievalRouterDecision(BaseModel):
    """LLM-generated retrieval execution plan
    
    Each query selects the most suitable strategy (4 types):
    1. hybrid: BM25+Semantic+RRF (tools layer fusion) - suitable for complex queries
    2. bm25: Pure keyword search - suitable for CPT codes, exact matching
    3. semantic: Pure semantic search - suitable for conceptual queries
    4. bm25_semantic: BM25 then Semantic, keeping both result sets - suitable for dual verification
    
    All query results are aggregated and ranked using _aggregate_and_rank
    """
    pre_filtering: PreFilteringConfig
    query_strategy_mapping: List[QueryStrategyMapping] = Field(
        min_length=1,
        description="Each query selects best strategy from: hybrid, bm25, semantic, bm25_semantic"
    )
    retrieval_parameters: RetrievalParameters
    fusion_parameters: FusionParameters
    reasoning: str = Field(min_length=50, max_length=500)


class PlanningRetrievalRouter:
    """
    Planning Mode - LLM generates plan, Agent executes
    
    Execution Flow:
    1. LLM generates RetrievalRouterDecision (Pydantic schema)
    2. Agent executes hardcoded plan:
       - Pre-filtering (range routing)
       - Each query selects most suitable strategy (hybrid/bm25/semantic/bm25_semantic)
       - All query results aggregated and ranked using _aggregate_and_rank
    3. Return Top-K results
    
    Design Principles:
    - Tools layer handles: single query retrieval + hybrid internal fusion (BM25+Semantic+RRF)
    - Planning layer handles: multi-query result aggregation (all strategy results ranked together)
    - Each query independently selects strategy based on characteristics (CPT codes use bm25, complex queries use hybrid, etc.)
    - bm25_semantic: one query uses both BM25 and Semantic, keeping both result sets
    """
    
    def __init__(self, config, tools, client=None):
        """
        Args:
            config: Configuration object with Azure OpenAI settings
            tools: RetrievalTools instance
            client: Azure OpenAI client (optional, will use config.client if not provided)
        """
        self.config = config
        self.tools = tools
        self.client = client if client is not None else getattr(config, 'client', None)
    
    def process(self, state: AgenticRAGState) -> dict:
        """
        LLM generates plan, Agent executes
        
        Supports retry mode: uses refined_queries when retry_count > 0
        
        Args:
            state: Contains question, query_candidates/refined_queries, retry_count, keep_chunks
            
        Returns:
            dict: Contains retrieved_chunks and retrieval_metadata
        """
        question = state["question"]
        question_type = state.get("question_type", "general")
        retrieval_strategies = state.get("retrieval_strategies", ["hybrid"])
        
        # RETRY MODE: Check if this is a retry round
        retry_count = state.get("retry_count", 0)
        
        # Use refined_queries if in retry mode, otherwise use query_candidates
        if retry_count > 0 and state.get("refined_queries"):
            # Retry mode: use refined queries from Query Refiner
            query_candidates = state.get("refined_queries", [])
            logger.info(f"\n🔄 RETRY MODE (Round {retry_count}) - Planning with {len(query_candidates)} refined queries")
            
            # Extract retrieval hints from refined queries (Query Refiner's hints)
            retrieval_hints = [
                rq.get("retrieval_hint") 
                for rq in query_candidates 
                if isinstance(rq, dict) and rq.get("retrieval_hint")
            ]
            if retrieval_hints:
                logger.info(f"   Using {len(retrieval_hints)} retrieval hints from Query Refiner")
        else:
            # Initial mode: use query candidates from Query Planner
            query_candidates = state.get("query_candidates", [])
            # Get retrieval hints from Query Planner (strategy-level recommendations)
            retrieval_hints = state.get("retrieval_hints", [])
        
        # Build prompt for LLM to generate plan
        prompt = build_retrieval_router_prompt(
            question=question,
            question_type=question_type,
            retrieval_strategies=retrieval_strategies,
            query_candidates=[
                {
                    "query": qc.get("query") if isinstance(qc, dict) else qc.query,
                    "query_type": qc.get("query_type") if isinstance(qc, dict) else qc.query_type,
                    "weight": qc.get("weight", 1.0) if isinstance(qc, dict) else qc.weight
                }
                for qc in query_candidates
            ],
            retrieval_hints=retrieval_hints
        )
        
        # Log prompt length for debugging
        prompt_length = len(prompt)
        logger.info(f"   📏 Prompt length: {prompt_length} characters (~{prompt_length//4} tokens)")
        if prompt_length > 20000:  # Warn if prompt is very long
            logger.warning(f"   ⚠️  Very long prompt detected! This may cause token limit issues.")
        
        # Call LLM for execution plan (only LLM call)
        logger.info(f"   📞 Calling LLM to generate retrieval plan...")
        
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.config.azure_deployment_name,
                messages=[
                    {"role": "system", "content": RETRIEVAL_ROUTER_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                ],
                response_format=RetrievalRouterDecision,
                temperature=self.config.agent_temperature,
                max_tokens=4000  # Enough for complex plans, prevents runaway generation
            )
            logger.info(f"   ✅ LLM plan generated successfully")
        except Exception as e:
            # Log the error with context
            logger.error(f"   ❌ LLM plan generation failed: {str(e)}")
            if "length" in str(e).lower() or "token" in str(e).lower():
                logger.error(f"   ⚠️  Token limit exceeded. Prompt tokens: Check if refined_queries are too long.")
            raise
        
        decision = response.choices[0].message.parsed
        
        # Execute based on LLM's plan (Agent executes hardcoded logic)
        return self._execute_plan(state, decision)
    
    def _execute_plan(self, state: AgenticRAGState, decision: RetrievalRouterDecision) -> dict:
        """
        Agent executes retrieval according to LLM's plan
        
        ⚠️ This is entirely Agent's hardcoded logic, no LLM involvement
        
        Args:
            state: Current state
            decision: LLM-generated execution plan
            
        Returns:
            dict: Retrieved chunks and metadata
        """
        all_results = []
        metadata = {
            "mode": "planning",
            "retry_count": state.get("retry_count", 0),  # Always include retry_count
            "plan_reasoning": decision.reasoning,
            "strategies_used": [],  # Strategy used for each query (will be set in Step 2)
            "total_chunks_retrieved": 0,
            "boost_range_results": decision.fusion_parameters.boost_range_results
        }
        
        # Step 1: Pre-filtering (Agent executes)
        range_chunk_ids = set()
        cpt_descriptions = {}  # Store CPT descriptions for query enhancement
        
        if decision.pre_filtering.apply_range_routing:
            # Convert codes to integers
            cpt_codes_list = [int(code) for code in decision.pre_filtering.range_filter_cpt_codes]
            
            # Batch get all CPT descriptions at once (more efficient)
            cpt_descriptions = self.tools.get_cpt_descriptions(cpt_codes_list)
            
            # Get range routing chunks for each code
            for code in cpt_codes_list:
                try:
                    chunk_ids = self.tools.range_routing(
                        code,
                        limit=decision.pre_filtering.range_filter_limit
                    )
                    range_chunk_ids.update(chunk_ids)
                except:
                    pass  # Skip invalid codes
            metadata["range_chunks_count"] = len(range_chunk_ids)
            metadata["cpt_descriptions_used"] = cpt_descriptions
        
        # Step 2: Execute retrieval - Each query uses its most suitable strategy (Agent executes)
        query_candidates = state.get("query_candidates", [])
        params = decision.retrieval_parameters
        strategies_used = []  # Record strategies used
        per_query_stats = []  # Track detailed execution info
        
        for mapping in decision.query_strategy_mapping:
            # Get corresponding query
            query = query_candidates[mapping.query_index]
            query_text = query.get("query") if isinstance(query, dict) else query.query
            query_weight = query.get("weight", 1.0) if isinstance(query, dict) else query.weight
            query_guidance = query.get("guidance") if isinstance(query, dict) else (query.guidance if hasattr(query, 'guidance') else None)
            strategy = mapping.strategy
            
            # Enhance query with CPT descriptions if available
            if cpt_descriptions:
                desc_text = " ".join(cpt_descriptions.values())
                query_text = f"{query_text} [CPT Description: {desc_text}]"
            
            # Initialize stats for this query
            query_stats = {
                "query_index": mapping.query_index,
                "query_text": query_text[:80] + "..." if len(query_text) > 80 else query_text,
                "strategy": strategy,
                "weight": query_weight,
                "tools_called": [],
                "chunks_retrieved": 0
            }
            
            # Execute retrieval based on each query's strategy
            if strategy == "hybrid":
                # hybrid: BM25+Semantic+RRF (fusion at tools layer)
                results = self.tools.hybrid_search(
                    query_text,
                    top_k=params.hybrid_top_k,
                    bm25_weight=params.hybrid_bm25_weight,
                    semantic_weight=params.hybrid_semantic_weight,
                    guidance=query_guidance
                )
                
                query_stats["tools_called"].append("hybrid_search")
                query_stats["chunks_retrieved"] = len(results)
                
                for r in results:
                    r.score *= query_weight
                    if r.chunk_id in range_chunk_ids:
                        r.score *= decision.fusion_parameters.boost_range_results
                
                all_results.extend(results)
                strategies_used.append(f"q{mapping.query_index}:hybrid")
            
            elif strategy == "bm25":
                # bm25: Pure keyword search
                results = self.tools.bm25_search(query_text, top_k=params.bm25_top_k)
                
                query_stats["tools_called"].append("bm25_search")
                query_stats["chunks_retrieved"] = len(results)
                
                for r in results:
                    r.score *= query_weight
                    if r.chunk_id in range_chunk_ids:
                        r.score *= decision.fusion_parameters.boost_range_results
                
                all_results.extend(results)
                strategies_used.append(f"q{mapping.query_index}:bm25")
            
            elif strategy == "semantic":
                # semantic: Pure semantic search
                results = self.tools.semantic_search(
                    query_text, 
                    top_k=params.semantic_top_k,
                    guidance=query_guidance
                )
                
                query_stats["tools_called"].append("semantic_search")
                query_stats["chunks_retrieved"] = len(results)
                
                for r in results:
                    r.score *= query_weight
                    if r.chunk_id in range_chunk_ids:
                        r.score *= decision.fusion_parameters.boost_range_results
                
                all_results.extend(results)
                strategies_used.append(f"q{mapping.query_index}:semantic")
            
            elif strategy == "bm25_semantic":
                # bm25_semantic: One query uses both BM25 and Semantic, keeping both result sets
                # BM25 retrieval
                bm25_results = self.tools.bm25_search(query_text, top_k=params.bm25_top_k)
                for r in bm25_results:
                    r.score *= query_weight
                    if r.chunk_id in range_chunk_ids:
                        r.score *= decision.fusion_parameters.boost_range_results
                all_results.extend(bm25_results)
                
                # Semantic retrieval
                semantic_results = self.tools.semantic_search(
                    query_text, 
                    top_k=params.semantic_top_k,
                    guidance=query_guidance
                )
                for r in semantic_results:
                    r.score *= query_weight
                    if r.chunk_id in range_chunk_ids:
                        r.score *= decision.fusion_parameters.boost_range_results
                all_results.extend(semantic_results)
                
                query_stats["tools_called"] = ["bm25_search", "semantic_search"]
                query_stats["chunks_retrieved"] = len(bm25_results) + len(semantic_results)
                
                strategies_used.append(f"q{mapping.query_index}:bm25_semantic")
            
            per_query_stats.append(query_stats)
        
        metadata["strategies_used"] = strategies_used  # Record strategy used for each query
        metadata["num_queries_executed"] = len(query_candidates)
        metadata["per_query_stats"] = per_query_stats  # Detailed execution info per query
        
        # Step 3: Fusion according to LLM's strategy (Agent executes)
        # Simplified: directly aggregate scores (weights and boost already applied in Step 2)
        aggregated_results = self._aggregate_and_rank(all_results)
        
        # Layer 2: Limit to reasonable number for Layer 3 (avoid sending too many to cross-encoder)
        # Planning mode typically retrieves more per query, so we limit here
        max_for_layer3 = 20  # Send at most 20 chunks to Layer 3 cross-encoder
        if len(aggregated_results) > max_for_layer3:
            aggregated_results = aggregated_results[:max_for_layer3]
            logger.info(f"\n📊 Layer 2: Truncated {len(all_results)} → {len(aggregated_results)} chunks for Layer 3")
        
        # RETRY MODE: Merge chunks if retry_count > 0
        retry_count = state.get("retry_count", 0)
        keep_chunks = state.get("keep_chunks", [])
        missing_aspects = state.get("missing_aspects", [])
        
        if retry_count > 0 and keep_chunks:
            logger.info(f"\n🔀 Merging chunks (Planning Mode - Round {retry_count}):")
            logger.info(f"   - Keep chunks (adaptive): {len(keep_chunks)}")
            logger.info(f"   - New chunks: {len(aggregated_results)}")
            logger.info(f"   - Missing aspects: {len(missing_aspects)}")
            
            # Call merge_chunks_in_retry tool
            # Return 15-20 merged chunks, then pass to Evidence Judge Layer 3 rerank
            merged_results, merge_stats = self.tools.merge_chunks_in_retry(
                keep_chunks=keep_chunks,
                new_chunks=aggregated_results,
                missing_aspects=missing_aspects,
                quality_threshold=0.75,
                top_k=20  # Return max 20 chunks, Evidence Judge will rerank to top 10
            )
            
            final_results = merged_results
            
            logger.info(f"\n✅ Merge complete (Adaptive Selection):")
            logger.info(f"   - Final chunks for Evidence Judge: {len(final_results)}")
            logger.info(f"   - Kept from old (adaptive): {merge_stats.get('kept_old_chunks', 0)}")
            logger.info(f"   - Added from new: {merge_stats.get('added_new_chunks', 0)}")
            logger.info(f"   - Removed duplicates: {merge_stats.get('duplicates_removed', 0)}")
            logger.info(f"   - Boosted for missing aspects: {merge_stats.get('boosted_chunks', 0)}")
            logger.info(f"   → Evidence Judge will rerank to top 10")
            
            # Update metadata with merge info
            metadata.update(merge_stats)
        else:
            # Initial retrieval (no merge)
            final_results = aggregated_results
        
        metadata["total_chunks_retrieved"] = len(final_results)
        metadata["unique_chunks_before_fusion"] = len(set(r.chunk_id for r in all_results))
        metadata["total_results_before_aggregation"] = len(all_results)
        
        # Save retrieved chunks to output
        from ..utils.save_workflow_outputs import save_retrieved_chunks
        saved_path = save_retrieved_chunks(
            chunks=final_results,
            question=state.get('question', ''),
            output_dir=self.config.retrieval_output_dir,
            metadata=metadata
        )
        metadata["saved_to"] = saved_path
        
        return {
            "retrieved_chunks": final_results,
            "retrieval_metadata": metadata,
            "cpt_descriptions": cpt_descriptions if cpt_descriptions else None
        }
    
    def _aggregate_and_rank(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Aggregate and rank retrieval results (Planning layer fusion)
        
        Handles two scenarios:
        1. Multi-query aggregation: Multiple queries retrieve same chunk → scores accumulate
        2. BM25+Semantic fusion: Under bm25_semantic strategy, BM25 and Semantic retrieve same chunk → scores accumulate
        
        Scores already applied in Step 2:
        - query_weight (from Query Planner)
        - range_boost (from LLM decision)
        
        Here we only need to:
        1. Deduplicate (same chunk may be retrieved by multiple queries or methods)
        2. Aggregate scores (sum)
        3. Sort
        
        Args:
            results: All retrieval results with weighted scores
            
        Returns:
            Deduplicated and ranked results
        """
        # Group by chunk_id and sum scores
        chunk_scores = {}
        chunk_data = {}
        
        for r in results:
            if r.chunk_id not in chunk_scores:
                chunk_scores[r.chunk_id] = 0.0
                chunk_data[r.chunk_id] = r
            
            # Sum scores (already includes query_weight and range_boost)
            chunk_scores[r.chunk_id] += r.score
        
        # Sort by aggregated score (descending)
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Convert back to RetrievalResult
        ranked_results = []
        for chunk_id, score in sorted_chunks:
            result = chunk_data[chunk_id]
            result.score = score  # Update with aggregated score
            ranked_results.append(result)
        
        return ranked_results
