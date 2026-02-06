"""
Planning Retrieval Router - LLM生成计划，Agent执行

特点：
- 1次LLM调用（生成plan）
- Agent按照plan硬编码执行
- 中等速度（~2秒）
- 中等成本（$0.01）
- LLM无法根据中间结果调整（plan已定）
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
    
    每个query根据其特点选择最适合的strategy：
    - hybrid: BM25+Semantic+RRF融合（适合复杂query）
    - bm25: 纯关键词检索（适合CPT codes、精确匹配）
    - semantic: 纯语义检索（适合概念性query）
    - bm25_semantic: 先BM25再Semantic，两组结果都保留（适合需要双重验证的query）
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
        default=2.0,  # ← 提高默认值到2.0
        ge=1.0,       # ← 最小1.0（不降分）
        le=5.0,       # ← 提高最大值到5.0，允许更强的boost
        description="Boost factor for range-routed chunks (higher = prioritize pre-filtered results more)"
    )


class RetrievalRouterDecision(BaseModel):
    """LLM生成的检索执行计划
    
    每个query根据特点选择最适合的strategy（4种）：
    1. hybrid: BM25+Semantic+RRF（tools层融合） - 适合复杂query
    2. bm25: 纯关键词检索 - 适合CPT codes、精确匹配
    3. semantic: 纯语义检索 - 适合概念性query
    4. bm25_semantic: 先BM25再Semantic，保留两组结果 - 适合需要双重验证
    
    所有query的结果最后用_aggregate_and_rank聚合排序
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
    Planning模式 - LLM生成计划，Agent执行
    
    执行流程：
    1. LLM生成RetrievalRouterDecision（Pydantic schema）
    2. Agent按照plan硬编码执行：
       - Pre-filtering (range routing)
       - 每个query选择最适合的strategy (hybrid/bm25/semantic/bm25_semantic)
       - 所有query的结果用_aggregate_and_rank聚合排序
    3. 返回Top-K结果
    
    设计原则：
    - Tools层负责：单query检索 + hybrid内部融合(BM25+Semantic+RRF)
    - Planning层负责：多query结果聚合 (所有strategies的结果一起排序)
    - 每个query独立选择strategy，根据query特点（CPT code用bm25，复杂query用hybrid等）
    - bm25_semantic: 一个query同时用BM25和Semantic，保留两组结果
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
        LLM生成计划，Agent执行
        
        Args:
            state: Contains question, question_type, retrieval_strategies, query_candidates
            
        Returns:
            dict: Contains retrieved_chunks and retrieval_metadata
        """
        question = state["question"]
        question_type = state.get("question_type", "general")
        retrieval_strategies = state.get("retrieval_strategies", ["hybrid"])
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
        
        # Call LLM for execution plan (唯一的LLM调用)
        response = self.client.beta.chat.completions.parse(
            model=self.config.azure_deployment_name,
            messages=[
                {"role": "system", "content": RETRIEVAL_ROUTER_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            response_format=RetrievalRouterDecision,
            temperature=self.config.agent_temperature
        )
        
        decision = response.choices[0].message.parsed
        
        # Execute based on LLM's plan (Agent硬编码执行)
        return self._execute_plan(state, decision)
    
    def _execute_plan(self, state: AgenticRAGState, decision: RetrievalRouterDecision) -> dict:
        """
        Agent按照LLM的plan执行检索
        
        ⚠️ 这里完全是Agent的硬编码逻辑，无LLM参与
        
        Args:
            state: Current state
            decision: LLM生成的执行计划
            
        Returns:
            dict: Retrieved chunks and metadata
        """
        all_results = []
        metadata = {
            "mode": "planning",
            "plan_reasoning": decision.reasoning,
            "strategies_used": [],  # 每个query用的strategy (will be set in Step 2)
            "total_chunks_retrieved": 0,
            "boost_range_results": decision.fusion_parameters.boost_range_results
        }
        
        # Step 1: Pre-filtering (Agent执行)
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
        
        # Step 2: Execute retrieval - 每个query用其最适合的strategy (Agent执行)
        query_candidates = state.get("query_candidates", [])
        params = decision.retrieval_parameters
        strategies_used = []  # 记录使用的strategies
        per_query_stats = []  # Track detailed execution info
        
        for mapping in decision.query_strategy_mapping:
            # 获取对应的query
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
            
            # 根据每个query的strategy执行对应的检索
            if strategy == "hybrid":
                # hybrid: BM25+Semantic+RRF (融合在tools层)
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
                # bm25: 纯关键词检索
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
                # semantic: 纯语义检索
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
                # bm25_semantic: 一个query同时用BM25和Semantic，保留两组结果
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
        
        metadata["strategies_used"] = strategies_used  # 记录每个query用的strategy
        metadata["num_queries_executed"] = len(query_candidates)
        metadata["per_query_stats"] = per_query_stats  # Detailed execution info per query
        
        # Step 3: Fusion按照LLM的策略 (Agent执行)
        # 简化：直接聚合scores（已经在Step 2中应用了weights和boost）
        final_results = self._aggregate_and_rank(all_results)
        
        # Take top_k
        final_results = final_results[:self.config.top_k]
        
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
            "retrieval_metadata": metadata
        }
    
    def _aggregate_and_rank(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        聚合和排序检索结果（Planning层的融合）
        
        处理两种情况：
        1. Multi-query aggregation: 多个queries检索到同一chunk → 分数累加
        2. BM25+Semantic fusion: bm25_semantic策略下，BM25和Semantic检索到同一chunk → 分数累加
        
        Scores已经在Step 2中应用了：
        - query_weight (来自 Query Planner)
        - range_boost (来自 LLM decision)
        
        这里只需要：
        1. 去重（同一chunk可能被多个queries或多个methods检索到）
        2. 聚合scores（累加）
        3. 排序
        
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
            
            # Sum scores (已经包含 query_weight 和 range_boost)
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
