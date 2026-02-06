"""
Retrieval tools for Agentic RAG system
Wraps existing BM25, Chroma, and hybrid retrieval

Note: All indices must be pre-built before using this class!
- BM25 index: Run build_bm25.py first
- Chroma vector DB: Run build_embeddings_chroma.py first
- Range index: Run build_range_index.py first
"""
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
from openai import AzureOpenAI

from .bm25_store import BM25Store
from .chroma_store import ChromaStore
from ..state import RetrievalResult, QueryCandidate
from ..config import AgenticRAGConfig


class RetrievalTools:
    """
    Retrieval tools that can be called by agents
    
    Prerequisites:
    1. BM25 index must exist at config.bm25_index_path
    2. ChromaDB must exist at config.chroma_db_path
    3. Range index must exist at config.range_index_path
    4. Chunks file must exist at config.chunks_path
    
    If any index is missing, initialization will fail with clear error message.
    """
    
    def __init__(self, config: AgenticRAGConfig):
        self.config = config
        
        # Validate all required files exist
        self._validate_indices()
        
        # Load retrieval indices
        self.bm25_store = BM25Store.load(config.bm25_index_path)
        self.chroma_store = ChromaStore(config.chroma_client, "ncci_chunks")
        
        # Load chunks map
        self.chunks_map = self._load_chunks_map(config.chunks_path)
        
        # Initialize embedding client (使用独立的embedding endpoint)
        self.embedding_client = AzureOpenAI(
            api_key=config.azure_openai_api_key_embedding,
            api_version=config.azure_api_version_embedding,
            azure_endpoint=config.azure_openai_endpoint_embedding,
        )
        self.embedding_deployment = config.azure_deployment_name_embedding
        
        # Initialize range index connection (keep open for performance)
        self.range_conn = sqlite3.connect(config.range_index_path)
        self.range_conn.row_factory = sqlite3.Row  # Enable column access by name
    
    def _validate_indices(self):
        """
        Validate that all required indices exist
        
        Raises:
            FileNotFoundError: If any required index is missing
        """
        required_files = {
            "BM25 index": self.config.bm25_index_path,
            "ChromaDB": self.config.chroma_db_path,
            "Range index": self.config.range_index_path,
            "Chunks file": self.config.chunks_path
        }
        
        missing_files = []
        for name, path in required_files.items():
            if not Path(path).exists():
                missing_files.append(f"{name}: {path}")
        
        if missing_files:
            error_msg = (
                "❌ Missing required indices! Please build them first:\n\n"
                + "\n".join(f"  - {f}" for f in missing_files)
                + "\n\nBuild commands:\n"
                + "  python src/tools/build_bm25.py\n"
                + "  python src/tools/build_embeddings_chroma.py\n"
                + "  python src/tools/build_range_index.py\n"
            )
            raise FileNotFoundError(error_msg)
    
    def __del__(self):
        """Close range index connection on cleanup"""
        if hasattr(self, 'range_conn'):
            self.range_conn.close()
    
    def _load_chunks_map(self, chunks_path: str) -> Dict[str, dict]:
        """Load chunks into memory"""
        chunks = {}
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                chunks[chunk["chunk_id"]] = chunk
        return chunks
    
    def _embed_query(self, text: str) -> List[float]:
        """Generate embedding for query text"""
        response = self.embedding_client.embeddings.create(
            model=self.config.azure_deployment_name_embedding,
            input=text
        )
        return response.data[0].embedding
    
    def range_routing(self, cpt_code: int, limit: int = 50) -> List[str]:
        """
        Range routing: lookup chunks by CPT code range
        
        Args:
            cpt_code: 5-digit CPT code (e.g., 14301)
            limit: Maximum number of chunk_ids to return
            
        Returns:
            List of chunk_ids, ordered by relevance weight (descending)
            
        Note: Uses persistent SQLite connection (opened during __init__)
        """
        cur = self.range_conn.cursor()
        cur.execute(
            "SELECT chunk_id, weight FROM range_index WHERE start <= ? AND end >= ? ORDER BY weight DESC LIMIT ?",
            (cpt_code, cpt_code, limit),
        )
        rows = cur.fetchall()
        
        return [row[0] for row in rows]
    
    def bm25_search(self, query: str, top_k: int = 50) -> List[RetrievalResult]:
        """
        BM25 lexical search
        """
        results = self.bm25_store.search(query, top_k=top_k)
        
        return [
            RetrievalResult(
                chunk_id=r["chunk_id"],
                score=r["score"],
                text=self.chunks_map.get(r["chunk_id"], {}).get("text", ""),
                metadata=self.chunks_map.get(r["chunk_id"], {}).get("metadata", {})
            )
            for r in results
        ]
    
    def semantic_search(self, query: str, top_k: int = 50) -> List[RetrievalResult]:
        """
        Semantic vector search using ChromaDB
        """
        query_embedding = self._embed_query(query)
        results = self.chroma_store.search(query_embedding, top_k=top_k)
        
        return [
            RetrievalResult(
                chunk_id=r["chunk_id"],
                score=r["score"],
                text=r["text"],
                metadata=r.get("metadata", {})
            )
            for r in results
        ]
    
    def hybrid_search(
        self, 
        query: str, 
        top_k: int = 20,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5
    ) -> List[RetrievalResult]:
        """
        Hybrid search with weighted RRF fusion
        Combines BM25 and semantic search results
        
        Args:
            query: Search query text
            top_k: Number of results to return
            bm25_weight: Weight for BM25 scores (default: 0.5)
            semantic_weight: Weight for semantic scores (default: 0.5)
            
        Note: Weights are applied to RRF scores before fusion
        """
        # Get BM25 results
        bm25_results = self.bm25_search(query, top_k=top_k * 2)
        
        # Get semantic results
        semantic_results = self.semantic_search(query, top_k=top_k * 2)
        
        # Weighted RRF fusion
        fused_scores = {}
        
        # Apply BM25 with weight
        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result.chunk_id
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + \
                bm25_weight * (1.0 / (self.config.rrf_k + rank))
        
        # Apply semantic with weight
        for rank, result in enumerate(semantic_results, start=1):
            chunk_id = result.chunk_id
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + \
                semantic_weight * (1.0 / (self.config.rrf_k + rank))
        
        # Sort by fused score and take top_k
        sorted_chunks = sorted(
            fused_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        # Convert back to RetrievalResult
        results = []
        for chunk_id, score in sorted_chunks:
            chunk_data = self.chunks_map.get(chunk_id, {})
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                score=score,
                text=chunk_data.get("text", ""),
                metadata=chunk_data.get("metadata", {})
            ))
        
        return results
    
    def _rrf_fuse(
        self, 
        *result_lists: List[RetrievalResult], 
        k: int = 60
    ) -> Dict[str, float]:
        """
        Reciprocal Rank Fusion
        Combines multiple ranked lists into a single score
        """
        scores = {}
        for result_list in result_lists:
            for rank, result in enumerate(result_list, start=1):
                chunk_id = result.chunk_id
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        return scores
    
    def multi_query_hybrid_search(
        self, 
        query_candidates: List[QueryCandidate], 
        cpt_codes: List[int] = None,
        top_k: int = 20
    ) -> tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Advanced retrieval with multiple queries and range routing
        
        This is an ORCHESTRATION-LEVEL tool that combines multiple retrieval steps.
        
        Workflow:
        1. Range routing with CPT codes (if provided) - pre-filter relevant chunks
        2. Execute hybrid search for each query candidate
        3. Weight results by candidate weight
        4. Fuse all results with RRF
        5. Boost range-matched chunks by 50%
        
        Args:
            query_candidates: List of QueryCandidate from Query Planner (typically 2-4)
            cpt_codes: List of CPT codes for range routing (optional, supports multiple codes)
            top_k: Number of final results to return
            
        Returns:
            (results, metadata) - Results list and retrieval statistics
            
        Usage by Mode:
            - Direct mode: ✅ ALWAYS uses this (primary use case)
              Query Planner generates candidates → Direct mode calls this function
            
            - Tool Calling mode: ❌ NOT exposed to LLM
              LLM only sees primitive tools (range_routing, bm25_search, etc.)
              This avoids LLM confusion between high-level vs low-level tools
            
            - Planning mode: ❌ NOT exposed to LLM
              Same reason as Tool Calling mode
              
        Design Philosophy:
            This is a "macro" that encapsulates best practices for multi-query retrieval.
            It should NOT be exposed to LLMs - they should compose primitives instead.
            Direct mode uses it to avoid code duplication and ensure consistency.
        """
        all_chunk_ids = set()
        retrieval_stats = {
            "range_routing_count": 0,
            "bm25_count": 0,
            "semantic_count": 0,
            "total_candidates": 0,
            "cpt_codes_used": cpt_codes if cpt_codes else []
        }
        
        # Step 1: Range routing if CPT codes provided (supports multiple codes)
        range_chunks = set()
        if cpt_codes:
            for cpt_code in cpt_codes:
                chunks = self.range_routing(cpt_code, limit=300)
                range_chunks.update(chunks)
            all_chunk_ids.update(range_chunks)
            retrieval_stats["range_routing_count"] = len(range_chunks)
        
        # Step 2: Execute multiple queries with hybrid search
        query_results = []
        for candidate in query_candidates:
            results = self.hybrid_search(candidate.query, top_k=top_k * 2)
            
            # Weight results by query candidate weight
            for r in results:
                r.score *= candidate.weight
            
            query_results.append(results)
        
        # Step 3: Fuse all query results
        all_results = []
        for results in query_results:
            all_results.extend(results)
        
        # Additional RRF fusion across queries
        fused_scores = self._rrf_fuse(*query_results)
        
        # Boost range routing chunks
        for chunk_id in range_chunks:
            if chunk_id in fused_scores:
                fused_scores[chunk_id] *= 1.5  # 50% boost for range matches
        
        # Sort and take top_k
        sorted_chunks = sorted(
            fused_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        # Convert to RetrievalResult
        final_results = []
        for chunk_id, score in sorted_chunks:
            chunk_data = self.chunks_map.get(chunk_id, {})
            final_results.append(RetrievalResult(
                chunk_id=chunk_id,
                score=score,
                text=chunk_data.get("text", ""),
                metadata=chunk_data.get("metadata", {})
            ))
        
        retrieval_stats["total_candidates"] = len(all_chunk_ids)
        retrieval_stats["final_count"] = len(final_results)
        
        return final_results, retrieval_stats
