"""
保存检索结果的辅助函数
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
    
    # 生成文件名（基于时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"retrieval_{timestamp}.json"
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
