"""
Agentic RAG Agents - Main Entry Point
===========================

This file serves as the coordinator and entry point for all Agents.
Specific Agent implementations are in the agents/ directory.

Architecture:
- agents/base.py - Base class
- agents/orchestrator.py - Decision Agent (question analysis + strategy hints)
- agents/query_planner.py - Query planning (used in planning mode)
- agents/retrieval_router.py - Retrieval routing (three modes)
  - retrieval_router_direct.py - Direct mode (0 LLM calls)
  - retrieval_router_planning.py - Planning mode (1 LLM call)
  - retrieval_router_tool_calling.py - Tool Calling mode (5-15 LLM calls)
- agents/evidence_judge.py - Evidence assessment (with new LLM summarization feature)
- agents/query_refiner.py - Query optimization (used during retry)
- agents/structured_extraction.py - Answer extraction

Key Design Decisions:
1. Retrieval Mode: Configuration-level choice (prod/dev/research), not per-query decision
2. Orchestrator Responsibility: Question analysis + Strategy hints (does not select mode)
3. Client Management: Unified use of config.client (lazy initialization pattern)
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
from src.agents.answer_generator import AnswerGeneratorAgent


class AgenticRAGAgents:
    """
    Agentic RAG Agents Coordinator
    
    This class is responsible for:
    1. Initializing all Agent instances
    2. Providing unified node interfaces for Workflow invocation
    3. Managing configuration sharing between Agents
    
    Usage:
        config = AgenticRAGConfig.from_env()
        agents = AgenticRAGAgents(config)
        tools = RetrievalTools(config)
        
        # Use in workflow
        result = agents.orchestrator_node(state)
    """
    
    def __init__(self, config: AgenticRAGConfig):
        """
        Initialize all Agents
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize each Agent (using lazy client from config)
        # Each agent internally gets the shared OpenAI client from config.client
        self.orchestrator = OrchestratorAgent(config)
        self.query_planner = QueryPlannerAgent(config)
        self.evidence_judge = EvidenceJudgeAgent(config)
        self.answer_generator = AnswerGeneratorAgent(config)  # NEW: Answer Generator
        self.query_refiner = QueryRefinerAgent(config)  # NEW: Query Refiner for retry (unified use of Azure OpenAI)
        # self.structured_extraction = StructuredExtractionAgent(config)
        
        # retrieval_router needs tools and mode passed in workflow
        self._retrieval_routers = {}  # Cache routers by mode
        # Cache Agent instances for different modes
        # self._retrieval_routers = {
        #     "direct": RetrievalRouterAgent(config, tools, mode="direct"),
        #     "planning": RetrievalRouterAgent(config, tools, mode="planning"),
        #     "tool_calling": RetrievalRouterAgent(config, tools, mode="tool_calling")
        # }
    
    # ========== Node Functions ==========
    # These functions are called by the workflow as LangGraph nodes
    
    def orchestrator_node(self, state: AgenticRAGState) -> dict:
        """
        Orchestrator node
        
        Responsibilities:
        1. Question type classification (cpt_code_lookup, billing_compatibility, etc.)
        2. Strategy hints (recommend which retrieval strategies to use)
        3. Max retry settings
        
        Note: Does not select retrieval mode (mode is a configuration-level decision)
        """
        return self.orchestrator.process(state)
    
    def query_planner_node(self, state: AgenticRAGState) -> dict:
        """
        Query Planner node
        
        Generate multiple query candidates (used in planning/tool_calling mode)
        """
        return self.query_planner.process(state)
    
    def retrieval_router_node(self, state: AgenticRAGState, tools, mode: str = None) -> dict:
        """
        Retrieval Router node
        
        Execute retrieval according to configured mode (not selected by orchestrator)
        
        Mode selection priority:
        1. Parameter mode (can be forcibly specified during testing)
        2. config.retrieval_mode (deployment environment configuration)
           - prod: "direct" (0 LLM calls, fastest)
           - dev: "planning" (1 LLM call, balanced)
           - research: "tool_calling" (5-15 LLM calls, smartest)
        
        Role of Orchestrator:
        - Provide strategy hints (recommend which retrieval strategies to use)
        - Different modes reference these hints:
          * Direct mode: Strictly execute hints
          * Planning mode: Reference hints to generate plan
          * Tool Calling mode: Hints as system message guidance
        
        Args:
            state: Current state (includes orchestrator's strategy hints)
            tools: RetrievalTools instance
            mode: Optional, overrides default mode in config
        """
        # Determine the mode to use (priority: parameter > config)
        if mode is None:
            mode = self.config.retrieval_mode
        
        # Lazy initialization of retrieval_router (cache one instance per mode)
        if mode not in self._retrieval_routers:
            self._retrieval_routers[mode] = RetrievalRouterAgent(
                self.config, 
                tools, 
                mode=mode
            )
        
        # Retrieval results (chunks), or when reusing Agent instance, new retrieval results
        result = self._retrieval_routers[mode].process(state)
        return result

    def evidence_judge_node(self, state: AgenticRAGState) -> dict:
        """
        Evidence Judge node
        
        Evaluate the sufficiency of retrieved evidence (using three-tier chunk formatting + LLM summarization)
        """
        return self.evidence_judge.process(state)
    
    def answer_generator_node(self, state: AgenticRAGState) -> dict:
        """
        Answer Generator node
        
        Generate final answer based on sufficient evidence (top 10 chunks)
        """
        return self.answer_generator.process(state)
    
    def query_refiner_node(self, state: AgenticRAGState) -> dict:
        """
        Query Refiner node
        
        Optimize queries to fill evidence gaps (used during retry)
        Generate refined queries and select keep_chunks
        """
        return self.query_refiner.process(state)
    
    # def structured_extraction_node(self, state: AgenticRAGState) -> dict:
    #     """
    #     Structured Extraction node
        
    #     Extract structured answers from evidence
    #     """
    #     return self.structured_extraction.process(state)