"""
Configuration for Agentic RAG system
"""
import os
from pathlib import Path
from pydantic import BaseModel
from typing import Optional
import dotenv
dotenv.load_dotenv()

class AgenticRAGConfig(BaseModel):
    """Configuration for Agentic RAG system"""
    
    # Project structure
    project_root: Path = Path(__file__).parent.parent  # agentic_rag/
    rag_root: Path = Path(__file__).parent.parent / "rag"  # agentic_rag/rag/
    
    # Data paths
    data_dir: str = "rag/data"
    raw_data_dir: str = "rag/data/raw"
    processed_data_dir: str = "rag/data/processed"
    
    # Build paths (indexes and intermediate files)
    build_dir: str = "rag/build"
    chunks_path: str = "rag/build/chunks.jsonl"
    range_index_path: str = "rag/build/cpt_range_index.db"
    bm25_index_path: str = "rag/build/bm25_index.pkl"
    chroma_db_path: str = "rag/build/chroma_db"
    
    # Output paths
    output_dir: str = "rag/output"
    query_output_dir: str = "rag/output/queries"
    eval_output_dir: str = "rag/output/evaluations"
    
    # Azure OpenAI settings
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_api_version: Optional[str] = None
    azure_deployment_name: Optional[str] = None  # For chat/completion
    
    # Azure OpenAI Embedding settings (可能使用不同的 endpoint)
    azure_openai_api_key_embedding: Optional[str] = None
    azure_openai_endpoint_embedding: Optional[str] = None
    azure_api_version_embedding: Optional[str] = None
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
        """
        Load configuration from environment variables
        """
        config = cls(
            # Azure OpenAI for chat/completion
            azure_openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_openai_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            azure_api_version=os.environ.get("AZURE_API_VERSION", "2024-12-01-preview"),
            azure_deployment_name=os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-4.1-mini"),
            
            # Azure OpenAI for embeddings (可能使用不同的 endpoint)
            azure_openai_api_key_embedding=os.environ.get(
                "AZURE_OPENAI_API_KEY_EMBEDDING",
                os.environ.get("AZURE_OPENAI_API_KEY")  # Fallback to main API key
            ),
            azure_openai_endpoint_embedding=os.environ.get(
                "AZURE_OPENAI_ENDPOINT_EMBEDDING",
                os.environ.get("AZURE_OPENAI_ENDPOINT")  # Fallback to main endpoint
            ),
            azure_api_version_embedding=os.environ.get(
                "AZURE_API_VERSION_EMBEDDING",
                os.environ.get("AZURE_API_VERSION", "2024-02-01")  # Fallback to main version
            ),
            azure_deployment_name_embedding=os.environ.get(
                "AZURE_DEPLOYMENT_NAME_EMBEDDING", 
                "text-embedding-3-large"
            ),
        )
        
        # 确保所有目录存在
        config.ensure_directories()
        
        return config
    
    def ensure_directories(self) -> None:
        """ensure all necessary directories exist"""
        directories = [
            self.get_path(self.data_dir),
            self.get_path(self.raw_data_dir),
            self.get_path(self.processed_data_dir),
            self.get_path(self.build_dir),
            self.get_path(self.output_dir),
            self.get_path(self.query_output_dir),
            self.get_path(self.eval_output_dir),
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_path(self, relative_path: str) -> Path:
        return self.project_root / relative_path
    
    def get_chunks_path(self) -> Path:
        return self.get_path(self.chunks_path)
    
    def get_bm25_index_path(self) -> Path:
        return self.get_path(self.bm25_index_path)
    
    def get_chroma_db_path(self) -> Path:
        return self.get_path(self.chroma_db_path)
