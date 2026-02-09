"""
Chunk Formatting Utilities

Reusable functions for formatting retrieval chunks for display to LLM
"""

from typing import List
from ..state import RetrievalResult


def format_chunks_with_ids(chunks: List[RetrievalResult]) -> str:
    """
    格式化 chunks，添加编号和 ID 标识（用于 Answer Generator）
    
    每个 chunk 显示为:
    ### Chunk [1] - chunk_000210 (Score: 0.xxxx) [CPT: xxxxx]
    {chunk text}
    
    编号 [1] [2] [3] 用于在答案中引用
    chunk_id 用于追溯原始数据
    
    Args:
        chunks: List of RetrievalResult objects
        
    Returns:
        Formatted string with numbered chunks and IDs
    """
    if not chunks:
        return "No chunks available."
    
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        # Extract chunk data (支持 dict 和 object 两种格式)
        if isinstance(chunk, dict):
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            text = chunk.get("text", "")
            score = chunk.get("score", 0.0)
            metadata = chunk.get("metadata", {})
        else:
            chunk_id = chunk.chunk_id
            text = chunk.text
            score = chunk.score
            metadata = chunk.metadata
        
        # Format metadata
        cpt_info = f" [CPT: {metadata.get('cpt_code')}]" if metadata.get('cpt_code') else ""
        
        # Format chunk with number [1] [2] [3] for citation
        formatted.append(
            f"### Chunk [{i}] - {chunk_id} (Score: {score:.4f}){cpt_info}\n"
            f"{text}\n"
        )
    
    return "\n---\n\n".join(formatted)


def format_cpt_descriptions(cpt_descriptions: dict) -> str:
    """
    格式化 CPT code descriptions
    
    Args:
        cpt_descriptions: Dict of CPT code -> description
        
    Returns:
        Formatted CPT descriptions section (or empty string if none)
    """
    if not cpt_descriptions:
        return ""
    
    desc_text = "### 📋 CPT Code Definitions\n\n"
    for code, description in cpt_descriptions.items():
        desc_text += f"**CPT {code}**: {description}\n\n"
    
    return desc_text
