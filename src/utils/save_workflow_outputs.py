"""
保存workflow各阶段输出的辅助函数

包含的输出类型：
- Retrieved chunks (检索到的文档块)
- Query candidates (生成的查询候选)
- 未来可扩展：Evaluations, Final answers等
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
    保存检索到的chunks到output目录
    
    Args:
        chunks: Retrieved chunks (list of RetrievalResult objects)
        question: Original question
        output_dir: Output directory path
        metadata: Additional metadata (mode, strategies, etc.)
        
    Returns:
        str: Saved file path
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 生成文件名（包含mode和时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = metadata.get('mode', 'unknown') if metadata else 'unknown'
    filename = f"retrieval_{mode}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # 准备保存的数据
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
    
    # 保存到文件
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
    保存query candidates (sub queries)到output/queries目录
    
    Args:
        query_candidates: List of QueryCandidate objects
        question: Original question
        output_dir: Output directory path
        metadata: Additional metadata (question_type, complexity, etc.)
        
    Returns:
        str: Saved file path
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 生成文件名（基于时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"queries_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # 准备保存的数据
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
    
    # 保存到文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return filepath
