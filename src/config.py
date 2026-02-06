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
    output_dir: str = "output"
    query_output_dir: str = "output/queries"
    eval_output_dir: str = "output/evaluations"
    retrieval_output_dir: str = "output/retrievals"  # Retrieved chunks
    
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
    retrieval_mode: str = "tool_calling"  # "direct" | "planning" | "tool_calling"
    top_k: int = 15
    max_retry: int = 2
    rrf_k: int = 60
    
    # Evidence quality thresholds
    min_coverage_score: float = 0.6
    min_specificity_score: float = 0.5
    min_citation_count: int = 3
    
    # Agent model settings
    agent_temperature: float = 0
    
    # Private client instances (lazy initialization)
    _client: Optional[object] = None
    _embedding_client: Optional[object] = None
    _chroma_client: Optional[object] = None
    
    @property
    def client(self):
        """
        Lazy initialization of shared Azure OpenAI client
        
        只在第一次访问config.client时创建，避免不必要的连接开销。
        所有agents共享这个client。
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
    def embedding_client(self):
        """
        Lazy initialization of Azure OpenAI embedding client
        
        用于生成embeddings，使用独立的endpoint配置。
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
        
        ChromaDB对同一路径只允许一个PersistentClient实例，
        所以必须共享这个client避免冲突。
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
        从环境变量加载配置（推荐方式）
        
        用法:
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
