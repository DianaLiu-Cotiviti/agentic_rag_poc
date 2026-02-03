"""
Base Agent class - 所有Agent的基类
提供共享的功能和接口
"""

from abc import ABC, abstractmethod
from openai import AzureOpenAI
from ..config import AgenticRAGConfig
from ..state import AgenticRAGState


class BaseAgent(ABC):
    """
    Agent基类
    
    所有Agent都继承这个基类，获得：
    1. LLM客户端
    2. 配置访问
    3. 标准化的接口
    """
    
    def __init__(self, config: AgenticRAGConfig):
        self.config = config
        self._client = None
    
    @property
    def client(self) -> AzureOpenAI:
        """延迟初始化的LLM客户端"""
        if self._client is None:
            self._client = AzureOpenAI(
                api_key=self.config.azure_openai_api_key,
                api_version=self.config.azure_api_version,
                azure_endpoint=self.config.azure_openai_endpoint
            )
        return self._client
    
    @abstractmethod
    def process(self, state: AgenticRAGState) -> dict:
        """
        处理state并返回更新
        
        每个Agent必须实现这个方法
        
        Args:
            state: 当前的RAG状态
        
        Returns:
            dict: 要更新到state中的内容
        """
        pass
    
    def __call__(self, state: AgenticRAGState) -> dict:
        """
        让Agent可以像函数一样调用
        
        用法:
            agent = OrchestratorAgent(config)
            result = agent(state)  # 等同于 agent.process(state)
        """
        return self.process(state)
