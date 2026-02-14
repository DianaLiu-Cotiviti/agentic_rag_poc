"""
Helper functions for saving workflow outputs at each stage

Output types included:
- Retrieved chunks (retrieved document chunks)
- Query candidates (generated query candidates)
- Final answers (Answer Generator final responses)
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


def save_retrieved_chunks(
    chunks: List[Any],
    question: str,
    output_dir: str = "output/retrievals",
    metadata: Dict[str, Any] = None
) -> str:
    """
    Save retrieved chunks to output directory
    
    Args:
        chunks: Retrieved chunks (list of RetrievalResult objects)
        question: Original question
        output_dir: Output directory path
        metadata: Additional metadata (mode, strategies, etc.)
        
    Returns:
        str: Saved file path
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate filename (including mode and timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = metadata.get('mode', 'unknown') if metadata else 'unknown'
    filename = f"retrieval_{mode}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Prepare data to save
    data = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "num_chunks": len(chunks),
        "metadata": metadata or {},
        "chunks": [
            {
                "chunk_id": getattr(chunk, 'chunk_id', ''),
                "score": getattr(chunk, 'score', 0.0),
                "text": getattr(chunk, 'text', ''),
                "metadata": getattr(chunk, 'metadata', {})
            }
            for chunk in chunks
        ]
    }
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return filepath


def save_query_candidates(
    query_candidates: List[Any],
    question: str,
    output_dir: str = "output/queries",
    metadata: Dict[str, Any] = None
) -> str:
    """
    Save query candidates (sub queries) to output/queries directory
    
    Args:
        query_candidates: List of QueryCandidate objects
        question: Original question
        output_dir: Output directory path
        metadata: Additional metadata (question_type, complexity, etc.)
        
    Returns:
        str: Saved file path
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate filename (based on timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"queries_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Prepare data to save
    data = {
        "timestamp": datetime.now().isoformat(),
        "original_question": question,
        "num_candidates": len(query_candidates),
        "metadata": metadata or {},
        "query_candidates": [
            {
                "query": getattr(qc, 'query', qc.get('query', '') if isinstance(qc, dict) else ''),
                "query_type": getattr(qc, 'query_type', qc.get('query_type', '') if isinstance(qc, dict) else ''),
                "weight": getattr(qc, 'weight', qc.get('weight', 1.0) if isinstance(qc, dict) else 1.0)
            }
            for qc in query_candidates
        ]
    }
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return filepath


def save_final_answer(
    final_answer: Dict[str, Any],
    question: str,
    output_dir: str = "output/responses",
    metadata: Dict[str, Any] = None
) -> str:
    """
    Save Answer Generator's final answer to output/responses directory
    
    Args:
        final_answer: CitedAnswer object (dict with answer, key_points, citations, etc.)
        question: Original question
        output_dir: Output directory path
        metadata: Additional metadata (mode, num_chunks, sufficiency, etc.)
        
    Returns:
        str: Saved file path
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate filename (based on mode and timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = metadata.get('mode', 'unknown') if metadata else 'unknown'
    filename = f"answer_{mode}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Prepare data to save
    data = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "metadata": metadata or {},
        "final_answer": final_answer
    }
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return filepath


def save_top10_chunks(
    top10_chunks: List[Any],
    question: str,
    output_dir: str = "output/retrievals",
    metadata: Dict[str, Any] = None
) -> str:
    """
    Save top 10 chunks after Evidence Judge Layer 3 reranking to output/retrievals directory
    These serve as direct evidence for LLM answer generation
    
    Args:
        top10_chunks: Top 10 chunks after cross-encoder reranking
        question: Original question
        output_dir: Output directory path
        metadata: Additional metadata (mode, original_count, etc.)
        
    Returns:
        str: Saved file path
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate filename (including mode and timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = metadata.get('mode', 'unknown') if metadata else 'unknown'
    filename = f"top10_chunks_{mode}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Prepare data to save
    data = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "num_chunks": len(top10_chunks),
        "metadata": metadata or {},
        "top10_chunks": [
            {
                "rank": i + 1,
                "chunk_id": getattr(chunk, 'chunk_id', ''),
                "score": getattr(chunk, 'score', 0.0),
                "text": getattr(chunk, 'text', ''),
                "metadata": getattr(chunk, 'metadata', {})
            }
            for i, chunk in enumerate(top10_chunks)
        ]
    }
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return filepath


def save_reranked_chunks(
    question: str,
    original_chunks: List[Any],
    reranked_chunks: List[Any],
    metadata: Dict[str, Any] = None,
    output_dir: str = "output/retrievals/layer3_reranking"
) -> str:
    """
    Save Layer 3 cross-encoder reranking results
    
    Used for analyzing and debugging reranking effectiveness:
    - Compare ranking changes before and after reranking
    - View CE scores vs original scores
    - Analyze which chunks were promoted/demoted/filtered
    
    Args:
        question: Original user question
        original_chunks: Chunks before reranking (15-20 from Layer 1-2)
        reranked_chunks: Chunks after reranking (top 10)
        metadata: Metadata including cross_encoder_model, mode, etc.
        output_dir: Output directory path
        
    Returns:
        str: Saved file path
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate filename (based on timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"layer3_rerank_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Calculate rank changes
    def calculate_rank_change(chunk_id: str, old_rank: int) -> str:
        """Calculate rank change"""
        for i, chunk in enumerate(reranked_chunks):
            cid = getattr(chunk, 'chunk_id', chunk.get('chunk_id', '') if isinstance(chunk, dict) else '')
            if cid == chunk_id:
                new_rank = i
                change = old_rank - new_rank
                return f"+{change}" if change > 0 else f"{change}" if change < 0 else "0"
        return "dropped"
    
    # Prepare data to save
    data = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "metadata": {
            "cross_encoder_model": metadata.get("cross_encoder_model", "unknown") if metadata else "unknown",
            "retrieval_mode": metadata.get("mode", "unknown") if metadata else "unknown",
            "chunks_before_rerank": len(original_chunks),
            "chunks_after_rerank": len(reranked_chunks),
            "chunks_dropped": len(original_chunks) - len(reranked_chunks)
        },
        "before_reranking": [
            {
                "rank": i + 1,
                "chunk_id": getattr(chunk, 'chunk_id', ''),
                "score": getattr(chunk, 'score', 0.0),
                "text_preview": (getattr(chunk, 'text', '')[:200] + "...") if len(getattr(chunk, 'text', '')) > 200 else getattr(chunk, 'text', ''),
                "cpt_code": getattr(chunk, 'metadata', {}).get('cpt_code'),
                "rank_change": calculate_rank_change(getattr(chunk, 'chunk_id', ''), i)
            }
            for i, chunk in enumerate(original_chunks)
        ],
        "after_reranking": [
            {
                "rank": i + 1,
                "chunk_id": getattr(chunk, 'chunk_id', ''),
                "ce_score": getattr(chunk, 'score', 0.0),
                "original_score": getattr(chunk, 'metadata', {}).get('original_score', 0.0),
                "text_preview": (getattr(chunk, 'text', '')[:200] + "...") if len(getattr(chunk, 'text', '')) > 200 else getattr(chunk, 'text', ''),
                "cpt_code": getattr(chunk, 'metadata', {}).get('cpt_code')
            }
            for i, chunk in enumerate(reranked_chunks)
        ],
        "score_statistics": {
            "original_score_range": f"{getattr(original_chunks[0], 'score', 0):.4f} → {getattr(original_chunks[-1], 'score', 0):.4f}" if original_chunks else "N/A",
            "ce_score_range": f"{getattr(reranked_chunks[0], 'score', 0):.4f} → {getattr(reranked_chunks[-1], 'score', 0):.4f}" if reranked_chunks else "N/A"
        }
    }
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return filepath
