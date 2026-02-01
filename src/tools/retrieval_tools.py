"""
Retrieval tools for Agentic RAG system
Wraps existing BM25, Chroma, and hybrid retrieval
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
    """Retrieval tools that can be called by agents"""
    
    def __init__(self, config: AgenticRAGConfig):
        self.config = config
        
        # Load retrieval indices
        self.bm25_store = BM25Store.load(config.bm25_index_path)
        self.chroma_store = ChromaStore(config.chroma_db_path, "ncci_chunks")
        
        # Load chunks map
        self.chunks_map = self._load_chunks_map(config.chunks_path)
        
        # Initialize embedding client
        self.embedding_client = AzureOpenAI(
            api_key=config.azure_openai_api_key,
            api_version=config.azure_api_version,
            azure_endpoint=config.azure_openai_endpoint,
        )
    
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
    
    def range_routing(self, cpt_code: int, limit: int = 300) -> List[str]:
        """
        Range routing: lookup chunks by CPT code range
        Returns list of chunk_ids
        """
        conn = sqlite3.connect(self.config.range_index_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT chunk_id, weight FROM range_index WHERE start <= ? AND end >= ? ORDER BY weight DESC LIMIT ?",
            (cpt_code, cpt_code, limit),
        )
        rows = cur.fetchall()
        conn.close()
        
        return [row[0] for row in rows]
    
    def bm25_search(self, query: str, top_k: int = 20) -> List[RetrievalResult]:
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
    
    def semantic_search(self, query: str, top_k: int = 20) -> List[RetrievalResult]:
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
        Hybrid search with RRF fusion
        Combines BM25 and semantic search results
        """
        # Get BM25 results
        bm25_results = self.bm25_search(query, top_k=top_k * 2)
        
        # Get semantic results
        semantic_results = self.semantic_search(query, top_k=top_k * 2)
        
        # RRF fusion
        fused_scores = self._rrf_fuse(
            bm25_results, 
            semantic_results, 
            k=self.config.rrf_k
        )
        
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
        cpt_code: int = None,
        top_k: int = 20
    ) -> tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Advanced retrieval with multiple queries and range routing
        
        Returns:
            (results, metadata)
        """
        all_chunk_ids = set()
        retrieval_stats = {
            "range_routing_count": 0,
            "bm25_count": 0,
            "semantic_count": 0,
            "total_candidates": 0
        }
        
        # Step 1: Range routing if CPT code provided
        range_chunks = set()
        if cpt_code:
            range_chunks = set(self.range_routing(cpt_code, limit=300))
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
