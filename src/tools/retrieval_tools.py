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
from ..state import RetrievalResult, QueryCandidate, SearchGuidance
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
        
        # Initialize embedding client (using separate embedding endpoint)
        self.embedding_client = AzureOpenAI(
            api_key=config.azure_openai_api_key_embedding,
            api_version=config.azure_api_version_embedding,
            azure_endpoint=config.azure_openai_endpoint_embedding,
        )
        self.embedding_deployment = config.azure_deployment_name_embedding
        
        # Initialize range index connection (keep open for performance)
        self.range_conn = sqlite3.connect(config.range_index_path)
        self.range_conn.row_factory = sqlite3.Row  # Enable column access by name
        
        # Initialize logger
        import logging
        self.logger = logging.getLogger("agenticrag.retrieval_tools")
    
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
    
    def get_cpt_description(self, cpt_code: int) -> str:
        """
        Get the full description for a single CPT/HCPCS code
        
        Args:
            cpt_code: CPT or HCPCS code (as integer or can be converted to string)
            
        Returns:
            Description string, or empty string if not found
            
        Note: Uses cached descriptions from config.cpt_descriptions
        """
        code_str = str(cpt_code)
        return self.config.cpt_descriptions.get(code_str, "")
    
    def get_cpt_descriptions(self, cpt_codes: List[int]) -> Dict[str, str]:
        """
        Get descriptions for multiple CPT/HCPCS codes at once (batch query)
        
        Args:
            cpt_codes: List of CPT/HCPCS codes
            
        Returns:
            Dict mapping code (as string) to description
            Only includes codes that were found (missing codes are omitted)
            
        Example:
            >>> tools.get_cpt_descriptions([14301, 27702, 99999])
            {
                "14301": "Adjacent tissue transfer...",
                "27702": "Arthroplasty, ankle..."
                # 99999 not found, so not included
            }
        """
        descriptions = {}
        for code in cpt_codes:
            code_str = str(code)
            if code_str in self.config.cpt_descriptions:
                descriptions[code_str] = self.config.cpt_descriptions[code_str]
        return descriptions
    
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
    
    def semantic_search(
        self, 
        query: str, 
        top_k: int = 50,
        guidance: SearchGuidance = None
    ) -> List[RetrievalResult]:
        """
        Semantic vector search using ChromaDB with optional search guidance
        
        Args:
            query: Search query text
            top_k: Number of results to return
            guidance: Optional SearchGuidance to enhance the query
            
        Returns:
            List of RetrievalResult objects
        """
        # Enhance query with guidance if provided
        enhanced_query = query
        if guidance and guidance.semantic_guidance:
            # Prepend guidance to query for better semantic matching
            enhanced_query = f"{guidance.semantic_guidance}\n\nQuery: {query}"
        
        query_embedding = self._embed_query(enhanced_query)
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
        semantic_weight: float = 0.5,
        guidance: SearchGuidance = None
    ) -> List[RetrievalResult]:
        """
        Hybrid search with weighted RRF fusion and optional search guidance
        Combines BM25 and semantic search results
        
        Args:
            query: Search query text
            top_k: Number of results to return
            bm25_weight: Weight for BM25 scores (default: 0.5)
            semantic_weight: Weight for semantic scores (default: 0.5)
            guidance: Optional SearchGuidance to enhance retrieval
            
        Returns:
            List of RetrievalResult objects
            
        Note: 
            - Weights are applied to RRF scores before fusion
            - Guidance is used to enhance semantic search with detailed instructions
            - BM25 search uses boost_terms from guidance if provided
        """
        # Enhance query with boost terms for BM25
        bm25_query = query
        if guidance and guidance.boost_terms:
            # Add boost terms to query for better keyword matching
            boost_str = " ".join(guidance.boost_terms)
            bm25_query = f"{query} {boost_str}"
        
        # Get BM25 results with potentially boosted query
        bm25_results = self.bm25_search(bm25_query, top_k=top_k * 2)
        
        # Get semantic results with guidance
        semantic_results = self.semantic_search(query, top_k=top_k * 2, guidance=guidance)
        
        # Layer 1 Reranking: Weighted RRF fusion (Method-level)
        # Purpose: Combine BM25 (keyword matching) + Semantic (concept matching)
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
        
        Steps:
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
        self.logger.info("🔍 Multi-Query Hybrid Search Execution Flow Trace")
        self.logger.info("—"*100)
        
        # Display input parameters
        self.logger.info(f"\n📥 Input Parameters:")
        self.logger.info(f"   - Query Candidates: {len(query_candidates)} queries")
        for idx, qc in enumerate(query_candidates, 1):
            has_guidance = hasattr(qc, 'guidance') and qc.guidance is not None
            self.logger.info(f"     {idx}. {qc.query[:60]}... (type={qc.query_type}, weight={qc.weight}, guidance={has_guidance})")
        self.logger.info(f"   - CPT Codes: {cpt_codes if cpt_codes else 'None'}")
        self.logger.info(f"   - Top K: {top_k}")
        
        all_chunk_ids = set()
        retrieval_stats = {
            "range_routing_count": 0,
            "bm25_count": 0,
            "semantic_count": 0,
            "total_candidates": 0,
            "cpt_codes_used": cpt_codes if cpt_codes else []
        }
        
        # Step 1: Range routing if CPT codes provided (supports multiple codes)
        self.logger.info(f"\n📌 Step 1: Range Routing (pre-filter relevant chunks)")
        range_chunks = set()
        if cpt_codes:
            self.logger.info(f"   CPT Codes: {cpt_codes}")
            for cpt_code in cpt_codes:
                chunks = self.range_routing(cpt_code, limit=300)
                self.logger.info(f"   - CPT {cpt_code}: found {len(chunks)} chunks")
                range_chunks.update(chunks)
            all_chunk_ids.update(range_chunks)
            retrieval_stats["range_routing_count"] = len(range_chunks)
            self.logger.info(f"   ✅ Range Routing Total: {len(range_chunks)} deduplicated chunks")
        else:
            self.logger.info(f"   ⏭️  No CPT codes, skipping Range Routing")
        
        # Step 2: Execute multiple queries with hybrid search using guidance
        self.logger.info(f"\n📌 Step 2: Hybrid Search (each query candidate enhanced with guidance)")
        self.logger.info(f"   Query Candidates count: {len(query_candidates)}")
        query_results = []
        for idx, candidate in enumerate(query_candidates, 1):
            # Use guidance if available
            guidance = candidate.guidance if hasattr(candidate, 'guidance') else None
            
            self.logger.info(f"\n   Query {idx}/{len(query_candidates)}:")
            self.logger.info(f"   - Query: {candidate.query[:80]}{'...' if len(candidate.query) > 80 else ''}")
            self.logger.info(f"   - Query Type: {candidate.query_type}")
            self.logger.info(f"   - Weight: {candidate.weight}")
            
            results = self.hybrid_search(
                candidate.query, 
                top_k=top_k * 2,
                guidance=guidance
            )
            
            self.logger.info(f"   ✅ Hybrid Search returned: {len(results)} chunks")
            if results:
                self.logger.info(f"      Top 3 scores (before weight): {[f'{r.score:.4f}' for r in results[:3]]}")
            
            # Weight results by query candidate weight
            for r in results:
                r.score *= candidate.weight
            
            if candidate.weight != 1.0 and results:
                self.logger.info(f"      Top 3 scores (after weight {candidate.weight}): {[f'{r.score:.4f}' for r in results[:3]]}")
            
            query_results.append(results)
        
        # Step 3: Fuse all query results
        self.logger.info(f"\n📌 Step 3: RRF Fusion (fuse results from all queries)")
        all_results = []
        for results in query_results:
            all_results.extend(results)
        
        self.logger.info(f"   Total chunks before merge: {len(all_results)} (including duplicates)")
        
        # Layer 2 Reranking: RRF fusion across queries (Query-level)
        # Purpose: Combine results from different sub-queries (multi-perspective)
        fused_scores = self._rrf_fuse(*query_results)
        self.logger.info(f"   Layer 2 RRF deduplicated chunks after fusion: {len(fused_scores)}")
        
        # Boost range routing chunks
        self.logger.info(f"\n📌 Step 4: Boost Range Routing Chunks (enhance pre-filtered chunks)")
        boosted_count = 0
        for chunk_id in range_chunks:
            if chunk_id in fused_scores:
                original_score = fused_scores[chunk_id]
                fused_scores[chunk_id] *= 1.5  # 50% boost for range matches
                boosted_count += 1
        
        if range_chunks:
            self.logger.info(f"   Range chunks: {len(range_chunks)}")
            self.logger.info(f"   Found and boosted in RRF results: {boosted_count} chunks (50% boost)")
            self.logger.info(f"   Range chunks not in RRF results: {len(range_chunks) - boosted_count}")
        else:
            self.logger.info(f"   ⏭️  No range chunks, skipping boost")
        
        # Sort and take top_k
        self.logger.info(f"\n📌 Step 5: Sort and Truncate (take top {top_k})")
        sorted_chunks = sorted(
            fused_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        self.logger.info(f"   Final return: {len(sorted_chunks)} chunks")
        if sorted_chunks:
            self.logger.info(f"   Top 5 final scores: {[f'{score:.4f}' for _, score in sorted_chunks[:5]]}")
            
            # Count how many in top_k come from range routing
            range_in_top = sum(1 for chunk_id, _ in sorted_chunks if chunk_id in range_chunks)
            if range_chunks:
                self.logger.info(f"   Top {top_k} from Range Routing: {range_in_top} ({range_in_top/len(sorted_chunks)*100:.1f}%)")
        
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
        
        self.logger.info(f"✅ Multi-Query Hybrid Search Complete")
        self.logger.info(f"   Statistics: {retrieval_stats}")
        self.logger.info(f"{'—'*100}\n")
        
        return final_results, retrieval_stats
    
    def cross_encoder_rerank(
        self,
        query: str,
        chunks: List[RetrievalResult],
        top_k: int = 10,
        model_name: str = None
    ) -> List[RetrievalResult]:
        """
        Layer 3 Reranking: Cross-Encoder reranks chunks
        
        Cross-encoder vs Bi-encoder:
        - Cross-encoder: query+doc encoded together, deep interaction (slow, suitable for reranking)
        - Bi-encoder: query and doc encoded separately, fast matching (fast, suitable for recall)
        
        Use case:
        - Layer 1-2 have already narrowed down to 15-20 candidate chunks
        - Evidence Judge or Answer Generator needs high-quality input
        
        Args:
            query: Original user question
            chunks: Retrieved chunks (typically 15-20 from Layer 1-2)
            top_k: Number of top chunks to return (default: 10)
            model_name: Cross-encoder model name (override config)
            
        Returns:
            Top-K reranked chunks with updated scores
            
        Example:
            >>> reranked = tools.cross_encoder_rerank(
            ...     query="What is CPT 14301?",
            ...     chunks=hybrid_results,
            ...     top_k=10
            ... )
        """
        try:
            from sentence_transformers import CrossEncoder
            import time
            
            # Use provided model or config default
            model = model_name or self.config.cross_encoder_model
            
            # Load model if not already loaded or if different model requested
            if not hasattr(self, '_cross_encoder') or self._cross_encoder_model != model:
                # Silently load cross-encoder model
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        self._cross_encoder = CrossEncoder(
                            model,
                            max_length=512,  # Explicit max length
                            device=None  # Auto-detect device (CPU/GPU)
                        )
                        self._cross_encoder_model = model
                        break
                    except Exception as load_error:
                        if attempt < max_retries - 1:
                            self.logger.info(f"   ⚠️  Load attempt {attempt + 1} failed: {load_error}")
                            self.logger.info(f"   🔄 Retrying in 2 seconds...")
                            time.sleep(2)
                        else:
                            raise
            
            # Prepare pairs for cross-encoder
            pairs = [(query, chunk.text) for chunk in chunks]
            
            # Batch predict scores
            ce_scores = self._cross_encoder.predict(pairs)
            
            # Combine chunks with CE scores
            chunk_score_pairs = list(zip(chunks, ce_scores))
            
            # Sort by cross-encoder score (descending)
            chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Keep top-K
            reranked_chunks = []
            for chunk, ce_score in chunk_score_pairs[:top_k]:
                # Create new RetrievalResult with updated score
                reranked_chunk = RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    score=float(ce_score),  # Replace with CE score
                    text=chunk.text,
                    metadata={
                        **chunk.metadata,
                        "original_score": chunk.score,  # Keep for comparison
                        "ce_score": float(ce_score),
                        "reranked_by": "cross_encoder"
                    }
                )
                reranked_chunks.append(reranked_chunk)
            
            # Statistics available but not logged to keep UI clean
            # CE Score range and top scores are computed during reranking
            
            return reranked_chunks
            
        except ImportError:
            self.logger.info("   ⚠️  sentence-transformers not installed, skipping reranking")
            self.logger.info("   💡 Install with: pip install sentence-transformers")
            return chunks[:top_k]
        except Exception as e:
            error_msg = str(e)
            self.logger.info(f"   ❌ Cross-encoder reranking failed: {error_msg}")
            
            # Provide helpful troubleshooting
            if "Can't load the model" in error_msg or "pytorch_model.bin" in error_msg:
                self.logger.info(f"   💡 Troubleshooting:")
                self.logger.info(f"      1. Model download may be incomplete - try running again")
                self.logger.info(f"      2. Check internet connection")
                self.logger.info(f"      3. Try alternative model: cross-encoder/ms-marco-MiniLM-L-6-v2 (smaller)")
                self.logger.info(f"      4. Set HF_TOKEN for faster downloads: export HF_TOKEN=your_token")
                self.logger.info(f"      5. Or disable cross-encoder in config: use_cross_encoder_rerank=False")
            
            self.logger.info(f"   ↩️  Falling back to original top-{top_k}")
            return chunks[:top_k]
    
    def merge_chunks_in_retry(
        self,
        keep_chunks: List[RetrievalResult],
        new_chunks: List[RetrievalResult],
        missing_aspects: List[str],
        quality_threshold: float = 0.75,
        top_k: int = 20
    ) -> tuple[List[RetrievalResult], dict]:
        """
        Intelligently merge old/new chunks (only used during retry) - OpenAI standard method
        
        Strategy:
        1. Keep old chunks (1-5 chunks, determined by adaptive selection)
        2. Deduplicate new chunks + quality threshold (score >= 0.75)
        3. Prioritize chunks that address missing_aspects (weighted 10%)
        4. RRF fusion
        5. Diversity filtering (max 3 chunks per document)
        
        Args:
            keep_chunks: Adaptively selected chunks from previous round (1-5 chunks)
            new_chunks: New chunks from current retrieval (15-20 chunks)
            missing_aspects: Missing aspects to prioritize
            quality_threshold: Minimum score for new chunks (default: 0.75)
            top_k: Maximum chunks to return (default: 20)
            
        Returns:
            Tuple of (merged_chunks, merge_stats)
        """
        self.logger.info(f"\n🔀 Merging Chunks (Retry Mode - OpenAI Strategy):")
        self.logger.info(f"   Old chunks (adaptive selection): {len(keep_chunks)}")
        self.logger.info(f"   New chunks (candidates): {len(new_chunks)}")
        
        # 1️⃣ Display kept old chunks (dynamic count: 1-5)
        self.logger.info(f"\n   📌 Keeping {len(keep_chunks)} high-quality chunks from previous round:")
        for i, chunk in enumerate(keep_chunks, 1):
            self.logger.info(f"      {i}. {chunk.chunk_id} (score: {chunk.score:.4f})")
        
        # 2️⃣ Filter new chunks: deduplicate + quality threshold
        qualified_new = []
        old_chunk_ids = {c.chunk_id for c in keep_chunks}
        
        for new_chunk in new_chunks:
            # Quality threshold
            if new_chunk.score < quality_threshold:
                continue
            
            # Deduplication check (simple: chunk_id)
            if new_chunk.chunk_id in old_chunk_ids:
                continue
            
            # Priority weighting: boost chunks that address missing_aspects
            if self._addresses_missing_aspect(new_chunk, missing_aspects):
                new_chunk.score *= 1.1  # Boost by 10%
                new_chunk._was_boosted = True  # Mark for stats
                self.logger.info(f"   ✨ Boosted {new_chunk.chunk_id} (addresses missing aspect)")
            
            qualified_new.append(new_chunk)
        
        self.logger.info(f"\n   ✅ Qualified new chunks: {len(qualified_new)}")
        self.logger.info(f"      (Filtered: score >= {quality_threshold}, no duplicates)")
        
        # 3️⃣ RRF fusion (based on chunk.score)
        all_chunks = keep_chunks + qualified_new
        merged = self._reciprocal_rank_fusion_merge(all_chunks)
        
        # 4️⃣ Diversity filtering (avoid all from same document)
        final = self._enforce_diversity(merged, max_per_doc=3, top_k=top_k)
        
        # Statistics
        kept_old_count = min(len(keep_chunks), len(final))
        new_in_final = len(final) - kept_old_count
        duplicates_removed = len(new_chunks) - len(qualified_new)
        boosted_count = sum(1 for c in qualified_new if hasattr(c, '_was_boosted'))
        
        self.logger.info(f"\n   🎯 Final merged chunks: {len(final)}")
        if final:
            self.logger.info(f"      Score range: {final[0].score:.4f} - {final[-1].score:.4f}")
        self.logger.info(f"      Composition: {kept_old_count} old + {new_in_final} new")
        
        merge_stats = {
            'kept_old_chunks': kept_old_count,
            'qualified_new_chunks': len(qualified_new),
            'added_new_chunks': new_in_final,
            'duplicates_removed': duplicates_removed,
            'boosted_chunks': boosted_count,
            'total_merged': len(final)
        }
        
        return final, merge_stats
    
    def _addresses_missing_aspect(self, chunk, missing_aspects):
        """Check if chunk addresses any missing aspect"""
        if not missing_aspects:
            return False
        
        chunk_text_lower = chunk.text.lower()
        for aspect in missing_aspects:
            # Simple keyword matching
            aspect_keywords = aspect.lower().split()
            if any(keyword in chunk_text_lower for keyword in aspect_keywords):
                return True
        return False
    
    def _reciprocal_rank_fusion_merge(self, chunks, k=60):
        """RRF fusion (based on chunk.score ranking)
        
        Returns chunks sorted by RRF score, with original score preserved
        """
        # Sort by score
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        
        # Calculate RRF score and store in dictionary (don't modify Pydantic object)
        rrf_scores = {}
        for rank, chunk in enumerate(sorted_chunks):
            rrf_score = 1.0 / (rank + k)
            rrf_scores[chunk.chunk_id] = rrf_score
        
        # Re-sort by RRF score (preserve original score for later cross-encoder)
        return sorted(sorted_chunks, key=lambda c: rrf_scores[c.chunk_id], reverse=True)
    
    def _enforce_diversity(self, chunks, max_per_doc=3, top_k=20):
        """Ensure diversity: max max_per_doc chunks per document"""
        doc_count = {}
        diverse_chunks = []
        
        for chunk in chunks:
            # Stop if we've reached top_k
            if len(diverse_chunks) >= top_k:
                break
                
            doc_id = chunk.metadata.get('document_id', chunk.metadata.get('source', 'unknown'))
            
            if doc_count.get(doc_id, 0) < max_per_doc:
                diverse_chunks.append(chunk)
                doc_count[doc_id] = doc_count.get(doc_id, 0) + 1
        
        return diverse_chunks
