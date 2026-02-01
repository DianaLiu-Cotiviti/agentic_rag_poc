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


def main():
    base_dir = "ncci_rag/" if os.path.exists("ncci_rag/build") else ""
    ncci_chunks_path = f"{base_dir}build/chunks.jsonl"
    chroma_db_path = f"{base_dir}build/chroma_db"  # Local persistence directory

    # Initialize Azure OpenAI
    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY_EMBEDDING"],
        api_version=os.environ["AZURE_API_VERSION_EMBEDDING"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_EMBEDDING"],
    )
    deployment = os.environ["AZURE_DEPLOYMENT_NAME_EMBEDDING"]

    # Initialize ChromaDB (persist to local directory)
    chroma_client = chromadb.PersistentClient(
        path=chroma_db_path,
        settings=Settings(
            anonymized_telemetry=False,  # Disable telemetry
            allow_reset=True
        )
    )
    
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
    chunks = load_chunks(ncci_chunks_path)
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


if __name__ == "__main__":
    main()
