"""
Unified Index Building Script
==============================

Automatically detects and builds all required indexes:
1. Range Index (SQLite)
2. BM25 Index (Pickle)
3. Embeddings Index (ChromaDB)

Skips if index already exists; automatically builds if missing.

Usage:
    from src.tools.build_indexes import ensure_all_indexes
    
    # Call before workflow starts
    ensure_all_indexes()
"""

import os
import sys
from pathlib import Path
from typing import Tuple


def check_range_index_exists(index_path: str) -> bool:
    """
    Check if Range Index exists and is valid
    
    Args:
        index_path: Range index database path
        
    Returns:
        bool: Whether it exists and is valid
    """
    if not os.path.exists(index_path):
        return False
    
    # Check file size (should be > 0)
    if os.path.getsize(index_path) == 0:
        return False
    
    # Check if it's a valid SQLite database
    try:
        import sqlite3
        conn = sqlite3.connect(index_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM range_index")
        count = cursor.fetchone()[0]
        conn.close()
        
        # Should have at least some records
        return count > 0
    except Exception:
        return False


def check_bm25_index_exists(index_path: str) -> bool:
    """
    Check if BM25 Index exists and is valid
    
    Args:
        index_path: BM25 index pickle file path
        
    Returns:
        bool: Whether it exists and is valid
    """
    if not os.path.exists(index_path):
        return False
    
    # Check file size (should be > 0)
    if os.path.getsize(index_path) == 0:
        return False
    
    # Try loading for validation
    try:
        import pickle
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
        # Check if required fields exist
        return 'bm25' in data and 'chunk_ids' in data
    except Exception:
        return False


def check_chroma_index_exists(chroma_dir: str) -> bool:
    """
    Check if ChromaDB Index exists and is valid
    
    Args:
        chroma_dir: ChromaDB storage directory
        
    Returns:
        bool: Whether it exists and is valid
    """
    if not os.path.exists(chroma_dir):
        return False
    
    # Check if chroma.sqlite3 file exists and has content
    # Avoid creating ChromaDB client (would cause multiple client conflicts)
    sqlite_path = os.path.join(chroma_dir, "chroma.sqlite3")
    if not os.path.exists(sqlite_path):
        return False
    
    # Check file size (should be > 0)
    if os.path.getsize(sqlite_path) == 0:
        return False
    
    # Simple check for vector segment directories (UUID-named directories)
    try:
        items = os.listdir(chroma_dir)
        # Should have at least chroma.sqlite3 and some UUID directories
        uuid_dirs = [item for item in items if len(item) == 36 and item.count('-') == 4]
        return len(uuid_dirs) > 0
    except Exception:
        return False


def build_range_index(chunks_path: str, index_path: str) -> None:
    """
    Build Range Index
    
    Args:
        chunks_path: chunks.jsonl file path
        index_path: Output database path
    """
    print("\n🔨 Building Range Index...")
    
    # Dynamically import build script
    from .build_range_index import build_range_db_index
    
    build_range_db_index(chunks_path, index_path)
    print(f"✅ Range Index built: {index_path}")


def build_bm25_index(chunks_path: str, index_path: str) -> None:
    """
    Build BM25 Index
    
    Args:
        chunks_path: chunks.jsonl file path
        index_path: Output pickle file path
    """
    print("\n🔨 Building BM25 Index...")
    
    import json
    from .bm25_store import BM25Store, tokenize
    from rank_bm25 import BM25Okapi
    
    texts = []
    chunk_ids = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_ids.append(c["chunk_id"])
            texts.append(c["text"])
    
    corpus = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(corpus)
    store = BM25Store(bm25=bm25, chunk_ids=chunk_ids)
    store.save(index_path)
    
    print(f"✅ BM25 Index built: {index_path} (docs={len(chunk_ids)})")


def build_chroma_index(chunks_path: str, chroma_dir: str, config=None) -> None:
    """
    Build ChromaDB Embeddings Index
    
    Args:
        chunks_path: chunks.jsonl file path
        chroma_dir: ChromaDB storage directory
        config: AgenticRAGConfig instance (optional)
    """
    print("\n🔨 Building ChromaDB Embeddings Index...")
    
    # Dynamically import build script
    from .build_embeddings_chroma import build_embeddings
    
    build_embeddings(chunks_path, chroma_dir, config=config)
    print(f"✅ ChromaDB Index built: {chroma_dir}")


def ensure_all_indexes(
    chunks_path: str = "rag/build/chunks.jsonl",
    range_index_path: str = "rag/build/cpt_range_index.db",
    bm25_index_path: str = "rag/build/bm25_index.pkl",
    chroma_dir: str = "rag/build/chroma_db",
    force_rebuild: bool = False,
    config=None
) -> Tuple[bool, bool, bool]:
    """
    Ensure all indexes are built
    
    Args:
        chunks_path: Chunks file path
        range_index_path: Range index database path
        bm25_index_path: BM25 index pickle path
        chroma_dir: ChromaDB storage directory
        force_rebuild: Whether to force rebuild all indexes
        config: AgenticRAGConfig instance (optional, for embedding client)
        
    Returns:
        Tuple[bool, bool, bool]: (range_built, bm25_built, chroma_built)
    """
    print("\n" + "="*80)
    print("📦 Checking and Building Indexes...")
    print("="*80)
    
    # Check if chunks file exists
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}\n"
            "Please ensure you have run the data preparation step first."
        )
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(range_index_path), exist_ok=True)
    os.makedirs(os.path.dirname(bm25_index_path), exist_ok=True)
    os.makedirs(chroma_dir, exist_ok=True)
    
    range_built = False
    bm25_built = False
    chroma_built = False
    
    # 1. Range Index
    print("\n📍 Checking Range Index...")
    if force_rebuild or not check_range_index_exists(range_index_path):
        build_range_index(chunks_path, range_index_path)
        range_built = True
    else:
        print(f"✓ Range Index already exists: {range_index_path}")
    
    # 2. BM25 Index
    print("\n📍 Checking BM25 Index...")
    if force_rebuild or not check_bm25_index_exists(bm25_index_path):
        build_bm25_index(chunks_path, bm25_index_path)
        bm25_built = True
    else:
        print(f"✓ BM25 Index already exists: {bm25_index_path}")
    
    # 3. ChromaDB Index
    print("\n📍 Checking ChromaDB Embeddings Index...")
    if force_rebuild or not check_chroma_index_exists(chroma_dir):
        build_chroma_index(chunks_path, chroma_dir, config=config)
        chroma_built = True
    else:
        print(f"✓ ChromaDB Index already exists: {chroma_dir}")
    
    print("\n" + "="*80)
    print("✅ All indexes ready!")
    print("="*80)
    
    return (range_built, bm25_built, chroma_built)

