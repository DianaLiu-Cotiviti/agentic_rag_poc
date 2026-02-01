"""
ChromaDB Store - Vector Retrieval Wrapper

Comparison with FAISS approach:
- Simpler: No need to manually manage numpy arrays and ID mappings
- Auto persistence: Directly saves to local directory
- Built-in metadata filtering: Can filter by chapter, section, etc.
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional


class ChromaStore:
    def __init__(self, persist_directory: str, collection_name: str = "ncci_chunks"):
        """
        Initialize ChromaDB
        
        Args:
            persist_directory: Local persistence directory (e.g., "build/chroma_db")
            collection_name: Collection name
        """
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_collection(name=collection_name)
    
    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 20,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Vector retrieval
        
        Args:
            query_embedding: Query vector (pass as list directly, no numpy needed!)
            top_k: Number of results to return
            where: Metadata filter condition (optional)
                   e.g., {"chapter": "Chapter XI"}
        
        Returns:
            [{"chunk_id": "chunk_001", "score": 0.92, "text": "...", "metadata": {...}}, ...]
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],  # ChromaDB requires list of embeddings
            n_results=top_k,
            where=where,  # Metadata filter
            include=["documents", "metadatas", "distances"]
        )
        
        # Format return results
        out = []
        for i in range(len(results['ids'][0])):
            out.append({
                "chunk_id": results['ids'][0][i],
                "score": 1.0 - results['distances'][0][i],  # Convert distance to similarity
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })
        return out
    
    def count(self) -> int:
        """Return the number of documents in the collection"""
        return self.collection.count()


# Usage example
if __name__ == "__main__":
    # Load ChromaDB
    store = ChromaStore(
        persist_directory="build/chroma_db",
        collection_name="ncci_chunks"
    )
    
    print(f"Total documents: {store.count()}")
    
    # Example query (need to generate query embedding first)
    # query_emb = [0.023, -0.045, ...]  # 3072 dimensions
    # results = store.search(query_emb, top_k=10)
    
    # Query with metadata filtering
    # results = store.search(
    #     query_emb, 
    #     top_k=10,
    #     where={"chapter": "Chapter XI"}  # Only search Chapter XI
    # )
