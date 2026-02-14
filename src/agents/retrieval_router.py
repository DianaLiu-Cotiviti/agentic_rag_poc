"""
Retrieval Router Agent - Unified retrieval routing entry point

Provides three modes:
1. Direct Mode - Fixed pipeline, 0 LLM calls (fastest, cheapest)
2. Tool Calling Mode - LLM-driven tool calling, 5-15 LLM calls (most intelligent)
3. Planning Mode - LLM generates plan, Agent executes, 1 LLM call (balanced)

Usage:
    # Direct mode (production, speed priority)
    router = RetrievalRouterAgent(config, tools, mode="direct")
    
    # Planning mode (standard scenario, balanced)
    router = RetrievalRouterAgent(config, tools, mode="planning")
    
    # Tool Calling mode (research environment, quality priority)
    router = RetrievalRouterAgent(config, tools, mode="tool_calling")
"""

from .base import BaseAgent
from ..state import AgenticRAGState
from .retrieval_router_direct import DirectRetrievalRouter
from .retrieval_router_tool_calling import ToolCallingRetrievalRouter
from .retrieval_router_planning import PlanningRetrievalRouter


class RetrievalRouterAgent(BaseAgent):
    """
    Retrieval Router Agent - Unified retrieval routing entry point
    
    Responsibilities:
    1. Select corresponding implementation based on mode (Direct/ToolCalling/Planning)
    2. Delegate retrieval execution to specific implementation
    
    Comparison of Three Modes:
    
    ┌──────────────┬──────────┬──────────┬────────┬──────────────┬─────────────┐
    │ Mode         │ LLM Calls│ Exec Time│  Cost  │ Intelligence │ Use Case    │
    ├──────────────┼──────────┼──────────┼────────┼──────────────┼─────────────┤
    │ direct       │   0      │ ~0.5s    │   $0   │    ⚡        │ Production  │
    │ planning     │   1      │  ~2s     │ $0.01  │  🤖🤖        │ Standard    │
    │ tool_calling │  5-15    │ ~10s     │ $0.05+ │ 🤖🤖🤖       │ Research    │
    └──────────────┴──────────┴──────────┴────────┴──────────────┴─────────────┘
    
    Implementation Details:
    - direct: See retrieval_router_direct.py
    - tool_calling: See retrieval_router_tool_calling.py  
    - planning: See retrieval_router_planning.py
    """
    
    def __init__(self, config, tools=None, mode="direct"):
        """
        Initialize retrieval router
        
        Args:
            config: Configuration object
            tools: RetrievalTools instance
            mode: "direct" | "tool_calling" | "planning"
        """
        super().__init__(config)
        self.tools = tools
        self.mode = mode
        
        # Initialize corresponding implementation based on mode
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
        Execute retrieval (delegate to specific implementation)
        
        Args:
            state: Contains retrieval_strategies, query_candidates, question_keywords
            
        Returns:
            dict: Contains retrieved_chunks and retrieval_metadata
        """
        return self.router.process(state)
