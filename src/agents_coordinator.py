"""
Agentic RAG Agents - 主入口
===========================

这个文件作为所有Agent的协调器和入口点。
具体的Agent实现都在 agents/ 目录下。

架构:
- agents/base.py - 基类
- agents/orchestrator.py - 决策Agent (question analysis + strategy hints)
- agents/query_planner.py - 查询规划 (planning mode使用)
- agents/retrieval_router.py - 检索路由 (三种模式)
  - retrieval_router_direct.py - Direct模式 (0 LLM调用)
  - retrieval_router_planning.py - Planning模式 (1 LLM调用)
  - retrieval_router_tool_calling.py - Tool Calling模式 (5-15 LLM调用)
- agents/evidence_judge.py - 证据评估 (新增LLM总结功能)
- agents/query_refiner.py - 查询优化 (retry时使用)
- agents/structured_extraction.py - 答案提取

关键设计决策:
1. Retrieval Mode: 配置级别选择（prod/dev/research），不是per-query决策
2. Orchestrator职责: Question analysis + Strategy hints（不选择mode）
3. Client管理: 统一使用config.client（lazy initialization pattern）
"""

from src.config import AgenticRAGConfig
from src.state import AgenticRAGState
from src.agents import (
    OrchestratorAgent,
    QueryPlannerAgent,
    EvidenceJudgeAgent,
    QueryRefinerAgent,
    StructuredExtractionAgent,
    RetrievalRouterAgent,
)


class AgenticRAGAgents:
    """
    Agentic RAG Agents 协调器
    
    这个类负责:
    1. 初始化所有Agent实例
    2. 提供统一的node接口给Workflow调用
    3. 管理Agent之间的配置共享
    
    用法:
        config = AgenticRAGConfig.from_env()
        agents = AgenticRAGAgents(config)
        tools = RetrievalTools(config)
        
        # 在workflow中使用
        result = agents.orchestrator_node(state)
    """
    
    def __init__(self, config: AgenticRAGConfig):
        """
        初始化所有Agent
        
        Args:
            config: 配置对象
        """
        self.config = config
        
        # 初始化各个Agent（使用config中的lazy client）
        # 每个agent内部会从config.client获取共享的OpenAI client
        self.orchestrator = OrchestratorAgent(config)
        self.query_planner = QueryPlannerAgent(config)
        self.evidence_judge = EvidenceJudgeAgent(config)
        # self.query_refiner = QueryRefinerAgent(config)
        # self.structured_extraction = StructuredExtractionAgent(config)
        
        # retrieval_router需要在workflow中传入tools和mode
        self._retrieval_routers = {}  # Cache routers by mode
        # 缓存不同mode的Agent实例
        # self._retrieval_routers = {
        #     "direct": RetrievalRouterAgent(config, tools, mode="direct"),
        #     "planning": RetrievalRouterAgent(config, tools, mode="planning"),
        #     "tool_calling": RetrievalRouterAgent(config, tools, mode="tool_calling")
        # }
    
    # ========== Node Functions ==========
    # 这些函数作为LangGraph节点被workflow调用
    
    def orchestrator_node(self, state: AgenticRAGState) -> dict:
        """
        Orchestrator节点
        
        职责:
        1. Question type classification (cpt_code_lookup, billing_compatibility, etc.)
        2. Strategy hints (建议使用哪些retrieval strategies)
        3. Max retry设置
        
        注意: 不选择retrieval mode（mode是配置级别的决策）
        """
        return self.orchestrator.process(state)
    
    def query_planner_node(self, state: AgenticRAGState) -> dict:
        """
        Query Planner节点
        
        生成多个查询候选（planning/tool_calling mode使用）
        """
        return self.query_planner.process(state)
    
    def retrieval_router_node(self, state: AgenticRAGState, tools, mode: str = None) -> dict:
        """
        Retrieval Router节点
        
        根据配置的mode执行检索（不是orchestrator选择）
        
        Mode选择优先级：
        1. 参数mode（测试时可强制指定）
        2. config.retrieval_mode（部署环境配置）
           - prod: "direct" (0 LLM calls, fastest)
           - dev: "planning" (1 LLM call, balanced)
           - research: "tool_calling" (5-15 LLM calls, smartest)
        
        Orchestrator的作用：
        - 提供strategy hints（建议使用哪些retrieval strategies）
        - 不同mode会参考这些hints：
          * Direct mode: 严格执行hints
          * Planning mode: 参考hints生成plan
          * Tool Calling mode: hints作为system message指导
        
        Args:
            state: 当前状态（包含orchestrator的strategy hints）
            tools: RetrievalTools实例
            mode: 可选，覆盖config中的默认mode
        """
        # 确定使用的模式（优先级：参数 > config）
        if mode is None:
            mode = self.config.retrieval_mode
        
        # 延迟初始化retrieval_router（每个mode缓存一个实例）
        if mode not in self._retrieval_routers:
            self._retrieval_routers[mode] = RetrievalRouterAgent(
                self.config, 
                tools, 
                mode=mode
            )
        
        # 检索结果 (chunks)，或者复用Agent instance时，新的检索结果
        result = self._retrieval_routers[mode].process(state)
        return result

    def evidence_judge_node(self, state: AgenticRAGState) -> dict:
        """
        Evidence Judge节点
        
        评估检索证据的充分性（使用三层chunk formatting + LLM总结）
        """
        return self.evidence_judge.process(state)
    
    # def query_refiner_node(self, state: AgenticRAGState) -> dict:
    #     """
    #     Query Refiner节点
        
    #     优化查询以填补证据缺失（retry时使用）
    #     """
    #     return self.query_refiner.process(state)
    
    # def structured_extraction_node(self, state: AgenticRAGState) -> dict:
    #     """
    #     Structured Extraction节点
        
    #     从证据中提取结构化答案
    #     """
    #     return self.structured_extraction.process(state)