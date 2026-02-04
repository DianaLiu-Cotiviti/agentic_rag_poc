"""
Direct Retrieval Router - 固定Pipeline执行

特点：
- 0次LLM调用
- 最快（~0.5秒）
- 最便宜（$0）
- 直接调用 multi_query_hybrid_search（无需重复实现）
"""

from typing import List, Dict, Any
from ..state import AgenticRAGState, RetrievalResult, QueryCandidate


class DirectRetrievalRouter:
    """
    Direct模式 - 固定Pipeline
    
    执行流程：
    1. 从 keywords 提取 CPT codes
    2. 直接调用 tools.multi_query_hybrid_search()
       - 内部自动处理: Range routing → Hybrid search → RRF fusion
    3. 返回结果
    
    Note: 不需要手动实现 pipeline，直接使用 RetrievalTools.multi_query_hybrid_search
    """
    
    def __init__(self, config, tools):
        """
        Args:
            config: Configuration object
            tools: RetrievalTools instance
        """
        self.config = config
        self.tools = tools
    
    def process(self, state: AgenticRAGState) -> dict:
        """
        直接执行检索（无LLM参与）
        
        直接调用 multi_query_hybrid_search，避免重复实现相同逻辑
        
        Args:
            state: Contains query_candidates, question_keywords
            
        Returns:
            dict: Contains retrieved_chunks and retrieval_metadata
        """
        query_candidates = state.get("query_candidates", [])
        question_keywords = state.get("question_keywords", [])
        
        # Extract CPT codes from keywords (supports multiple codes)
        from ..utils.keyword_parser import extract_cpt_codes, has_cpt_codes
        
        cpt_codes_str = extract_cpt_codes(question_keywords)
        cpt_codes = [int(code) for code in cpt_codes_str] if cpt_codes_str else None
        
        # Convert query_candidates to QueryCandidate objects if needed
        candidates = []
        for qc in query_candidates:
            if isinstance(qc, dict):
                candidates.append(QueryCandidate(
                    query=qc["query"],
                    query_type=qc.get("query_type", "original"),  # Default to "original" (valid Literal value)
                    weight=qc.get("weight", 1.0)
                ))
            else:
                candidates.append(qc)
        
        # Call multi_query_hybrid_search directly
        # This handles: Range routing (all CPT codes) → Hybrid search for all queries → RRF fusion → Boost
        results, retrieval_stats = self.tools.multi_query_hybrid_search(
            query_candidates=candidates,
            cpt_codes=cpt_codes,
            top_k=self.config.top_k
        )
        
        # Build actual strategies used (Direct mode = fixed pipeline)
        strategies_used = []
        if cpt_codes:
            strategies_used.append("range_routing")
        strategies_used.extend(["hybrid", "rrf_fusion"])
        
        # Add mode info to metadata
        metadata = {
            "mode": "direct",
            "strategies_used": strategies_used,
            "num_queries": len(candidates),
            **retrieval_stats
        }
        
        return {
            "retrieved_chunks": results,
            "retrieval_metadata": metadata
        }

