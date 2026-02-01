"""
Configuration for Agentic RAG system
"""
import os
from pathlib import Path
from pydantic import BaseModel
from typing import Optional


class AgenticRAGConfig(BaseModel):
    """Configuration for Agentic RAG system"""
    
    # Paths
    base_dir: str = ""
    chunks_path: str = "build/chunks.jsonl"
    range_index_path: str = "build/cpt_range_index.db"
    bm25_index_path: str = "build/bm25_index.pkl"
    chroma_db_path: str = "build/chroma_db"
    
    # Azure OpenAI settings
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_api_version: Optional[str] = None
    azure_deployment_name: Optional[str] = None  # For chat/completion
    azure_deployment_name_embedding: Optional[str] = None
    
    # Retrieval settings
    top_k: int = 15
    max_retry: int = 2
    rrf_k: int = 60
    
    # Evidence quality thresholds
    min_coverage_score: float = 0.6
    min_specificity_score: float = 0.5
    min_citation_count: int = 3
    
    # Agent model settings
    agent_temperature: float = 0.1
    agent_max_tokens: int = 2000
    
    @classmethod
    def from_env(cls) -> "AgenticRAGConfig":
        """Load configuration from environment variables"""
        base_dir = "ncci_rag/" if os.path.exists("ncci_rag/build") else ""
        
        return cls(
            base_dir=base_dir,
            chunks_path=f"{base_dir}build/chunks.jsonl",
            range_index_path=f"{base_dir}build/cpt_range_index.db",
            bm25_index_path=f"{base_dir}build/bm25_index.pkl",
            chroma_db_path=f"{base_dir}build/chroma_db",
            
            azure_openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY_EMBEDDING"),
            azure_openai_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT_EMBEDDING"),
            azure_api_version=os.environ.get("AZURE_API_VERSION_EMBEDDING", "2024-02-15-preview"),
            azure_deployment_name=os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-4o"),
            azure_deployment_name_embedding=os.environ.get("AZURE_DEPLOYMENT_NAME_EMBEDDING", "text-embedding-3-large"),
        )

    def get_full_path(self, relative_path: str) -> Path:
        """Get full path for a relative path"""
        return Path(self.base_dir) / relative_path if self.base_dir else Path(relative_path)
