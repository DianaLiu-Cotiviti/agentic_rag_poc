"""
Unified Index Building Script
==============================

自动检测并构建所有需要的索引：
1. Range Index (SQLite)
2. BM25 Index (Pickle)
3. Embeddings Index (ChromaDB)

如果index已存在则跳过，如果不存在则自动构建。

用法:
    from src.tools.build_indexes import ensure_all_indexes
    
    # 在workflow开始前调用
    ensure_all_indexes()
"""

import os
import sys
from pathlib import Path
from typing import Tuple


def check_range_index_exists(index_path: str) -> bool:
    """
    检查Range Index是否存在且有效
    
    Args:
        index_path: Range index数据库路径
        
    Returns:
        bool: 是否存在且有效
    """
    if not os.path.exists(index_path):
        return False
    
    # 检查文件大小（应该大于0）
    if os.path.getsize(index_path) == 0:
        return False
    
    # 检查是否是有效的SQLite数据库
    try:
        import sqlite3
        conn = sqlite3.connect(index_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM range_index")
        count = cursor.fetchone()[0]
        conn.close()
        
        # 至少应该有一些记录
        return count > 0
    except Exception:
        return False


def check_bm25_index_exists(index_path: str) -> bool:
    """
    检查BM25 Index是否存在且有效
    
    Args:
        index_path: BM25 index pickle文件路径
        
    Returns:
        bool: 是否存在且有效
    """
    if not os.path.exists(index_path):
        return False
    
    # 检查文件大小（应该大于0）
    if os.path.getsize(index_path) == 0:
        return False
    
    # 尝试加载验证
    try:
        import pickle
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
        # 检查是否有必要的字段
        return 'bm25' in data and 'chunk_ids' in data
    except Exception:
        return False


def check_chroma_index_exists(chroma_dir: str) -> bool:
    """
    检查ChromaDB Index是否存在且有效
    
    Args:
        chroma_dir: ChromaDB存储目录
        
    Returns:
        bool: 是否存在且有效
    """
    if not os.path.exists(chroma_dir):
        return False
    
    # 检查chroma.sqlite3文件是否存在且有内容
    # 避免创建ChromaDB client（会导致多个client冲突）
    sqlite_path = os.path.join(chroma_dir, "chroma.sqlite3")
    if not os.path.exists(sqlite_path):
        return False
    
    # 检查文件大小（应该大于0）
    if os.path.getsize(sqlite_path) == 0:
        return False
    
    # 简单检查是否有vector segment目录（UUID命名的目录）
    try:
        items = os.listdir(chroma_dir)
        # 至少应该有chroma.sqlite3和一些UUID目录
        uuid_dirs = [item for item in items if len(item) == 36 and item.count('-') == 4]
        return len(uuid_dirs) > 0
    except Exception:
        return False


def build_range_index(chunks_path: str, index_path: str) -> None:
    """
    构建Range Index
    
    Args:
        chunks_path: chunks.jsonl文件路径
        index_path: 输出的数据库路径
    """
    print("\n🔨 Building Range Index...")
    
    # 动态导入build脚本
    from .build_range_index import build_range_db_index
    
    build_range_db_index(chunks_path, index_path)
    print(f"✅ Range Index built: {index_path}")


def build_bm25_index(chunks_path: str, index_path: str) -> None:
    """
    构建BM25 Index
    
    Args:
        chunks_path: chunks.jsonl文件路径
        index_path: 输出的pickle文件路径
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
    构建ChromaDB Embeddings Index
    
    Args:
        chunks_path: chunks.jsonl文件路径
        chroma_dir: ChromaDB存储目录
        config: AgenticRAGConfig实例（可选）
    """
    print("\n🔨 Building ChromaDB Embeddings Index...")
    
    # 动态导入build脚本
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
    确保所有index都已构建
    
    Args:
        chunks_path: Chunks文件路径
        range_index_path: Range index数据库路径
        bm25_index_path: BM25 index pickle路径
        chroma_dir: ChromaDB存储目录
        force_rebuild: 是否强制重建所有index
        config: AgenticRAGConfig实例（可选，用于embedding client）
        
    Returns:
        Tuple[bool, bool, bool]: (range_built, bm25_built, chroma_built)
    """
    print("\n" + "="*80)
    print("📦 Checking and Building Indexes...")
    print("="*80)
    
    # 检查chunks文件是否存在
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}\n"
            "Please ensure you have run the data preparation step first."
        )
    
    # 确保输出目录存在
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

