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
    
    # CPT Code Description data
    cpt_description_path: str = "rag/data/raw/CPT Codes with Long Descriptions 2026.xlsx"
    
    # Output paths
    output_dir: str = "output"
    query_output_dir: str = "output/queries"
    eval_output_dir: str = "output/evaluations"
    retrieval_output_dir: str = "output/retrievals"  # Retrieved chunks
    response_output_dir: str = "output/responses"  # Final answers from Answer Generator
    
    # Memory paths
    memory_dir: str = "memory"  # Workflow execution history (project root level)
    
    # ========== Azure OpenAI Settings ==========
    # Chat/Completion (for all agents)
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_api_version: Optional[str] = None
    azure_deployment_name: Optional[str] = None
    
    # Embedding (for semantic search) - separate endpoint
    azure_openai_api_key_embedding: Optional[str] = None
    azure_openai_endpoint_embedding: Optional[str] = None
    azure_api_version_embedding: Optional[str] = None
    azure_deployment_name_embedding: Optional[str] = None
    
    # Retrieval settings
    retrieval_mode: str = "direct"  # "direct" | "planning" | "tool_calling"
    top_k: int = 15
    max_retry: int = 2
    rrf_k: int = 60
    
    # Evidence quality thresholds
    min_coverage_score: float = 0.6
    min_specificity_score: float = 0.5
    
    # Cross-encoder reranking (before Evidence Judge)
    use_cross_encoder_rerank: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Smaller, more reliable
    cross_encoder_top_k: int = 10  # Keep top-K after reranking
    
    # Agent model settings
    agent_temperature: float = 0
    
    # Private client instances (lazy initialization)
    _client: Optional[object] = None
    _embedding_client: Optional[object] = None
    _chroma_client: Optional[object] = None
    _cpt_descriptions: Optional[dict] = None  # CPT code -> description mapping
    
    @property
    def client(self):
        """
        Lazy initialization of shared Azure OpenAI client
        
        Created only on first access to config.client to avoid unnecessary connection overhead.
        All agents share this client instance.
        """
        if self._client is None:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                api_key=self.azure_openai_api_key,
                api_version=self.azure_api_version,
                azure_endpoint=self.azure_openai_endpoint
            )
        return self._client
    
    @property
    def cpt_descriptions(self) -> dict:
        """
        Lazy loading of CPT code descriptions
        
        Loaded from Excel file on first access, then cached for subsequent use.
        
        Returns:
            dict: {code: description} mapping for CPT and HCPCS codes
        """
        if self._cpt_descriptions is None:
            from .tools.preprocessing import load_cpt_descriptions
            self._cpt_descriptions = load_cpt_descriptions(self.cpt_description_path)
        return self._cpt_descriptions
    
    @property
    def embedding_client(self):
        """
        Lazy initialization of Azure OpenAI embedding client
        
        Used for generating embeddings with independent endpoint configuration.
        """
        if self._embedding_client is None:
            from openai import AzureOpenAI
            self._embedding_client = AzureOpenAI(
                api_key=self.azure_openai_api_key_embedding,
                api_version=self.azure_api_version_embedding,
                azure_endpoint=self.azure_openai_endpoint_embedding
            )
        return self._embedding_client
    
    @property
    def chroma_client(self):
        """
        Lazy initialization of shared ChromaDB client
        
        ChromaDB only allows one PersistentClient instance per path,
        so this client must be shared to avoid conflicts.
        """
        if self._chroma_client is None:
            import chromadb
            from chromadb.config import Settings
            self._chroma_client = chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        return self._chroma_client
    
    agent_max_tokens: int = 2000
    
    @classmethod
    def from_env(cls) -> "AgenticRAGConfig":
        """
        Load configuration from environment variables (recommended approach)
        
        Usage:
            config = AgenticRAGConfig.from_env()
            client = config.client  # Lazy initialization
        """
        config = cls(
            # Azure OpenAI (chat/completion)
            azure_openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_openai_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            azure_api_version=os.environ.get("AZURE_API_VERSION", "2024-12-01-preview"),
            azure_deployment_name=os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-4.1-mini"),
            
            # Embedding (separate endpoint/key)
            azure_openai_api_key_embedding=os.environ.get(
                "AZURE_OPENAI_API_KEY_EMBEDDING",
                os.environ.get("AZURE_OPENAI_API_KEY")  # Fallback to main key
            ),
            azure_openai_endpoint_embedding=os.environ.get(
                "AZURE_OPENAI_ENDPOINT_EMBEDDING",
                os.environ.get("AZURE_OPENAI_ENDPOINT")  # Fallback to main endpoint
            ),
            azure_api_version_embedding=os.environ.get(
                "AZURE_API_VERSION_EMBEDDING",
                os.environ.get("AZURE_API_VERSION", "2024-12-01-preview")  # Fallback
            ),
            azure_deployment_name_embedding=os.environ.get(
                "AZURE_DEPLOYMENT_NAME_EMBEDDING", 
                "text-embedding-3-large"
            ),
        )
        
        # Ensure all directories exist
        config.ensure_directories()
        
        return config
    
    def ensure_directories(self) -> None:
        """ensure all necessary directories exist"""
        directories = [
            self.get_path(self.data_dir),
            self.get_path(self.raw_data_dir),
            self.get_path(self.processed_data_dir),
            self.get_path(self.build_dir),
            self.get_path(self.memory_dir),  # Memory directory
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
