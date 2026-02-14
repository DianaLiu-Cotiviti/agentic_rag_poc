"""
Direct Retrieval Router - Fixed Pipeline Execution

Features:
- 0 LLM calls
- Fastest (~0.5 seconds)
- Cheapest ($0)
- Directly calls multi_query_hybrid_search (no need for duplicate implementation)
"""

from typing import List, Dict, Any
from ..state import AgenticRAGState, RetrievalResult, QueryCandidate


class DirectRetrievalRouter:
    """
    Direct Mode - Fixed Pipeline
    
    Execution Flow:
    1. Extract CPT codes from keywords
    2. Directly call tools.multi_query_hybrid_search()
       - Automatically handles internally: Range routing → Hybrid search → RRF fusion
    3. Return results
    
    Note: No need to manually implement pipeline, directly use RetrievalTools.multi_query_hybrid_search
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
        Execute retrieval directly (no LLM involvement)
        
        Directly call multi_query_hybrid_search to avoid duplicate implementation of same logic
        
        Note: Direct mode does not support retry (always generates answer directly)
        
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
        # Convert to List[int], or None if no codes found (empty list → None for cleaner logic)
        cpt_codes = [int(code) for code in cpt_codes_str] if cpt_codes_str else None
        
        # Batch get CPT descriptions for all codes at once (more efficient)
        cpt_descriptions = {}
        if cpt_codes:
            cpt_descriptions = self.tools.get_cpt_descriptions(cpt_codes)
        
        # Convert query_candidates to QueryCandidate objects if needed
        # Enhance queries with CPT descriptions for better semantic search
        candidates = []
        for qc in query_candidates:
            if isinstance(qc, dict):
                query_text = qc.get("query", "")
                # Enhance query with CPT descriptions if available
                if cpt_descriptions:
                    desc_text = " ".join(cpt_descriptions.values())
                    query_text = f"{query_text} [CPT Description: {desc_text}]"
                
                candidates.append(QueryCandidate(
                    query=query_text,
                    query_type=qc.get("query_type", qc.get("target_aspect", "refined")),
                    weight=qc.get("weight", 1.0),
                    guidance=qc.get("guidance", qc.get("retrieval_hint"))
                ))
            else:
                # Enhance existing QueryCandidate
                query_text = qc.query if hasattr(qc, 'query') else str(qc)
                if cpt_descriptions:
                    desc_text = " ".join(cpt_descriptions.values())
                    query_text = f"{query_text} [CPT Description: {desc_text}]"
                
                candidates.append(QueryCandidate(
                    query=query_text,
                    query_type=getattr(qc, 'query_type', 'refined') if hasattr(qc, 'query_type') else 'refined',
                    weight=getattr(qc, 'weight', 1.0) if hasattr(qc, 'weight') else 1.0,
                    guidance=getattr(qc, 'guidance', None) if hasattr(qc, 'guidance') else None
                ))
        
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
            "cpt_descriptions_used": cpt_descriptions,
            **retrieval_stats
        }
        
        # Save retrieved chunks to output
        from ..utils.save_workflow_outputs import save_retrieved_chunks
        saved_path = save_retrieved_chunks(
            chunks=results,
            question=state.get('question', ''),
            output_dir=self.config.retrieval_output_dir,
            metadata=metadata
        )
        metadata["saved_to"] = saved_path
        
        return {
            "retrieved_chunks": results,
            "retrieval_metadata": metadata,
            "cpt_descriptions": cpt_descriptions if cpt_descriptions else None
        }

