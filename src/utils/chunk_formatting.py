# Helper: get chunk text by id from a list of chunk dicts/objects
def get_chunk_text_by_id(chunks, chunk_id):
    for chunk in chunks:
        # Try both 'chunk_id' and 'id' for maximum compatibility
        if isinstance(chunk, dict):
            cid = chunk.get('chunk_id') or chunk.get('id')
            ctext = chunk.get('text')
        else:
            cid = getattr(chunk, 'chunk_id', None) or getattr(chunk, 'id', None)
            ctext = getattr(chunk, 'text', None)
        if cid == chunk_id:
            return ctext
    return None
"""
Chunk Formatting Utilities

Reusable functions for formatting retrieval chunks for display to LLM
"""

from typing import List, Dict, Any
from ..state import RetrievalResult


def format_chunks_with_ids(chunks: List[RetrievalResult]) -> str:
    """
    Format chunks with numbering and ID markers (for Answer Generator)
    
    Each chunk is displayed as:
    ### Chunk [1] - chunk_000210 (Score: 0.xxxx) [CPT: xxxxx]
    {chunk text}
    
    Numbers [1] [2] [3] are used for citation in answers
    chunk_id is used to trace back to original data
    
    Args:
        chunks: List of RetrievalResult objects
        
    Returns:
        Formatted string with numbered chunks and IDs
    """
    if not chunks:
        return "No chunks available."
    
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        # Extract chunk data (supports both dict and object formats)
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
    Format CPT code descriptions
    
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


def format_chunks_for_judge(chunks: List[RetrievalResult], cpt_descriptions: Dict[int, str] = None) -> str:
    """
    Format chunks for Evidence Judge evaluation
    
    Args:
        chunks: List of RetrievalResult objects
        cpt_descriptions: Optional CPT code descriptions
        
    Returns:
        Formatted string with all chunks
    """
    if not chunks:
        return "No chunks retrieved."
    
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        chunk_text = f"### Chunk {i}"
        if hasattr(chunk, 'score'):
            chunk_text += f" (Score: {chunk.score:.4f})"
        chunk_text += f"\n\n{chunk.text}\n"
        
        # Add metadata if available
        if hasattr(chunk, 'metadata') and chunk.metadata:
            metadata_items = []
            for key, value in chunk.metadata.items():
                if key not in ['text', 'embedding']:  # Skip large fields
                    metadata_items.append(f"{key}: {value}")
            if metadata_items:
                chunk_text += f"\n**Metadata**: {', '.join(metadata_items)}\n"
        
        formatted.append(chunk_text)
    
    # Add CPT descriptions at the top if available
    header = ""
    if cpt_descriptions:
        header = "**CPT Code Descriptions**:\n"
        for code, desc in cpt_descriptions.items():
            header += f"- CPT {code}: {desc}\n"
        header += "\n---\n\n"
    
    return header + "\n".join(formatted)
