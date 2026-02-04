"""
Retrieval Router Agent - 统一的检索路由入口

提供三种模式：
1. Direct Mode - 固定pipeline，0次LLM调用（最快，最便宜）
2. Tool Calling Mode - LLM驱动工具调用，5-15次LLM调用（最智能）
3. Planning Mode - LLM生成计划，Agent执行，1次LLM调用（平衡）

使用方法：
    # Direct模式（生产环境，速度优先）
    router = RetrievalRouterAgent(config, tools, mode="direct")
    
    # Planning模式（标准场景，平衡）
    router = RetrievalRouterAgent(config, tools, mode="planning")
    
    # Tool Calling模式（研究环境，质量优先）
    router = RetrievalRouterAgent(config, tools, mode="tool_calling")
"""

from .base import BaseAgent
from ..state import AgenticRAGState
from .retrieval_router_direct import DirectRetrievalRouter
from .retrieval_router_tool_calling import ToolCallingRetrievalRouter
from .retrieval_router_planning import PlanningRetrievalRouter


class RetrievalRouterAgent(BaseAgent):
    """
    Retrieval Router Agent - 统一的检索路由入口
    
    职责：
    1. 根据mode选择对应的实现（Direct/ToolCalling/Planning）
    2. 委托给具体实现执行检索
    
    三种模式对比：
    
    ┌──────────────┬──────────┬─────────┬────────┬──────────┬──────────┐
    │ 模式         │ LLM调用  │ 执行时间│  成本  │ 智能程度 │ 适用场景 │
    ├──────────────┼──────────┼─────────┼────────┼──────────┼──────────┤
    │ direct       │   0次    │ ~0.5秒  │   $0   │    ⚡    │ 生产环境 │
    │ planning     │   1次    │  ~2秒   │ $0.01  │  🤖🤖    │ 标准场景 │
    │ tool_calling │  5-15次  │ ~10秒   │ $0.05+ │ 🤖🤖🤖   │ 研究环境 │
    └──────────────┴──────────┴─────────┴────────┴──────────┴──────────┘
    
    实现细节：
    - direct: 见 retrieval_router_direct.py
    - tool_calling: 见 retrieval_router_tool_calling.py  
    - planning: 见 retrieval_router_planning.py
    """
    
    def __init__(self, config, tools=None, mode="direct"):
        """
        初始化检索路由器
        
        Args:
            config: Configuration object
            tools: RetrievalTools instance
            mode: "direct" | "tool_calling" | "planning"
        """
        super().__init__(config)
        self.tools = tools
        self.mode = mode
        
        # 根据模式初始化对应的实现
        if mode == "direct":
            self.router = DirectRetrievalRouter(config, tools)
        elif mode == "tool_calling":
            self.router = ToolCallingRetrievalRouter(config, tools)
        elif mode == "planning":
            self.router = PlanningRetrievalRouter(config, tools)
        else:
            raise ValueError(
                f"Invalid mode: {mode}. "
                f"Must be 'direct', 'tool_calling', or 'planning'"
            )
    
    def process(self, state: AgenticRAGState) -> dict:
        """
        执行检索（委托给具体实现）
        
        Args:
            state: Contains retrieval_strategies, query_candidates, question_keywords
            
        Returns:
            dict: Contains retrieved_chunks and retrieval_metadata
        """
        return self.router.process(state)
