# python ncci_rag/src/build_embeddings_chroma.py
"""
Build vector index using ChromaDB (recommended approach)

Advantages:
1. No manual NumPy Array conversion required
2. Automatic persistence to local directory
3. Built-in metadata filtering support
4. Simpler API

Installation: pip install chromadb
"""
import os
import json
import time
from tqdm import tqdm
from openai import AzureOpenAI
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ===== Configuration =====
BATCH_SIZE = 100        # Number of texts per batch
SLEEP_TIME = 0.5        # Sleep time between batches (seconds)
COLLECTION_NAME = "ncci_chunks"  # ChromaDB collection name


def load_chunks(path: str):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_embeddings(chunks_path: str, chroma_db_path: str, config=None):
    """
    构建ChromaDB Embeddings Index
    
    Args:
        chunks_path: chunks.jsonl文件路径
        chroma_db_path: ChromaDB存储目录
        config: AgenticRAGConfig实例（可选，如果不提供则从环境变量加载）
    """
    # Load config if not provided
    if config is None:
        from ..config import AgenticRAGConfig
        config = AgenticRAGConfig.from_env()
    
    # Use config's embedding client (consistent with agents)
    client = config.embedding_client
    deployment = config.azure_deployment_name_embedding

    # Use config's shared ChromaDB client
    chroma_client = config.chroma_client
    
    # Create or get collection
    # Delete if already exists
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except:
        pass
    
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )

    # Load chunks
    chunks = load_chunks(chunks_path)
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]
    
    # Prepare metadata (optional, for filtering)
    metadatas = []
    for c in chunks:
        metadatas.append({
            "chapter": c.get("chapter", ""),
            "section": c.get("section", ""),
            "page_start": c.get("page_start", 0),
            "page_end": c.get("page_end", 0),
        })
    
    print(f"Total chunks to process: {len(texts)}")
    print(f"Batch size: {BATCH_SIZE}, Sleep time: {SLEEP_TIME}s")
    print(f"Estimated time: {len(texts) / BATCH_SIZE * SLEEP_TIME / 60:.1f} minutes")
    print(f"ChromaDB path: {chroma_db_path}")

    # Generate embeddings in batches and add to ChromaDB
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Generating embeddings"):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_ids = chunk_ids[i:i + BATCH_SIZE]
        batch_metadatas = metadatas[i:i + BATCH_SIZE]
        
        try:
            # Generate embeddings
            resp = client.embeddings.create(model=deployment, input=batch_texts)
            batch_embeddings = [d.embedding for d in resp.data]
            
            # Add directly to ChromaDB (no NumPy conversion needed!)
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,  # Pass list[list[float]] directly
                documents=batch_texts,
                metadatas=batch_metadatas
            )
            
        except Exception as e:
            print(f"\n⚠ Error at batch {i//BATCH_SIZE + 1}: {e}")
            print(f"Retrying in 5 seconds...")
            time.sleep(5)
            resp = client.embeddings.create(model=deployment, input=batch_texts)
            batch_embeddings = [d.embedding for d in resp.data]
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
        
        time.sleep(SLEEP_TIME)

    print(f"\n✅ ChromaDB saved to: {chroma_db_path}")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Total documents: {collection.count()}")


def main():
    from ..config import AgenticRAGConfig
    
    config = AgenticRAGConfig.from_env()
    build_embeddings(config.chunks_path, config.chroma_db_path, config=config)


if __name__ == "__main__":
    main()
