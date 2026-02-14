"""
Base Agent class
Provides shared functionality and interfaces
"""

from abc import ABC, abstractmethod
from openai import AzureOpenAI
from ..config import AgenticRAGConfig
from ..state import AgenticRAGState


class BaseAgent(ABC):
    """
    Base Agent class
    
    All agents inherit this class to get:
    1. Shared LLM client (via config.client)
    2. Configuration access
    3. Standardized interface
    """
    
    def __init__(self, config: AgenticRAGConfig):
        self.config = config
    
    @property
    def client(self) -> AzureOpenAI:
        """
        Return shared LLM client
        
        All agents share config.client to avoid creating multiple connections
        """
        return self.config.client
    
    @abstractmethod
    def process(self, state: AgenticRAGState) -> dict:
        """
        Process state and return updates
        
        Each Agent must implement this method
        
        Args:
            state: Current RAG state
        
        Returns:
            dict: Content to update into state
        """
        pass
    
    def __call__(self, state: AgenticRAGState) -> dict:
        """
        Allows Agent to be called like a function
        
        Usage:
            agent = OrchestratorAgent(config)
            result = agent(state)  # Equivalent to agent.process(state)
        """
        return self.process(state)
