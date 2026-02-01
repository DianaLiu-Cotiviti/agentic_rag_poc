"""
LangGraph workflow definition for Agentic RAG
Orchestrates the multi-agent retrieval and reasoning pipeline
"""
from typing import Literal
from langgraph.graph import StateGraph, END

from .state import AgenticRAGState
from .agents import AgenticRAGAgents
from .tools.retrieval_tools import RetrievalTools
from .config import AgenticRAGConfig


class AgenticRAGWorkflow:
    """
    Agentic RAG workflow using LangGraph
    
    Workflow:
    1. Orchestrator → classify question & select strategy
    2. Query Planner → generate query candidates
    3. Retrieval Router → execute retrieval tools
    4. Evidence Judge → assess quality
    5. [Optional] Query Refiner → retry if insufficient
    6. Structured Extraction → final answer
    """
    
    def __init__(self, config: AgenticRAGConfig = None):
        self.config = config or AgenticRAGConfig.from_env()
        self.agents = AgenticRAGAgents(self.config)
        self.tools = RetrievalTools(self.config)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create graph
        workflow = StateGraph(AgenticRAGState)
        
        # Add nodes
        workflow.add_node("orchestrator", self._orchestrator_node)
        workflow.add_node("query_planner", self._query_planner_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("evidence_judge", self._evidence_judge_node)
        workflow.add_node("query_refiner", self._query_refiner_node)
        workflow.add_node("structured_extraction", self._structured_extraction_node)
        
        # Set entry point
        workflow.set_entry_point("orchestrator")
        
        # Add edges
        workflow.add_edge("orchestrator", "query_planner")
        workflow.add_edge("query_planner", "retrieval")
        workflow.add_edge("retrieval", "evidence_judge")
        
        # Conditional edge: retry or proceed to extraction
        workflow.add_conditional_edges(
            "evidence_judge",
            self._should_retry,
            {
                "retry": "query_refiner",
                "extract": "structured_extraction"
            }
        )
        
        workflow.add_edge("query_refiner", "retrieval")
        workflow.add_edge("structured_extraction", END)
        
        return workflow.compile()
    
    # ========== Node Functions ==========
    
    def _orchestrator_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Orchestrator node"""
        result = self.agents.orchestrator_node(state)
        state.update(result)
        return state
    
    def _query_planner_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Query Planner node"""
        result = self.agents.query_planner_node(state)
        state.update(result)
        return state
    
    def _retrieval_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """
        Retrieval Router & Execution node
        Delegates to retrieval_router_node agent
        """
        result = self.agents.retrieval_router_node(state, self.tools)
        state.update(result)
        return state
    
    def _evidence_judge_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Evidence Judge node"""
        result = self.agents.evidence_judge_node(state)
        state.update(result)
        return state
    
    def _query_refiner_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Query Refiner node"""
        result = self.agents.query_refiner_node(state)
        state.update(result)
        
        # Increment retry count
        state["retry_count"] = state.get("retry_count", 0) + 1
        
        return state
    
    def _structured_extraction_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Structured Extraction node"""
        result = self.agents.structured_extraction_node(state)
        state.update(result)
        return state
    
    # ========== Conditional Logic ==========
    
    def _should_retry(self, state: AgenticRAGState) -> Literal["retry", "extract"]:
        """
        Decide whether to retry with refined queries or proceed to extraction
        
        Decision Logic:
        1. Check if max retries reached (from orchestrator)
        2. Check evidence sufficiency (from evidence judge)
        3. Retry if: insufficient evidence AND retries available
        4. Extract if: sufficient evidence OR max retries reached
        
        NOTE: This function should NOT modify state (used for conditional routing)
        """
        assessment = state.get("evidence_assessment")
        retry_count = state.get("retry_count", 0)
        max_retry_allowed = state.get("max_retry_allowed", self.config.max_retry)
        
        # Safety: if no assessment, proceed to extraction
        if not assessment:
            return "extract"
        
        # If max retries reached, must proceed to extraction
        if retry_count >= max_retry_allowed:
            return "extract"
        
        # If evidence is insufficient and retries available, retry
        if not assessment.is_sufficient:
            return "retry"
        
        # Evidence is sufficient, proceed to extraction
        return "extract"
    
    # ========== Public Interface ==========
    
    def run(self, question: str, cpt_code: int = None, context: str = None) -> AgenticRAGState:
        """
        Run the Agentic RAG workflow
        
        Args:
            question: User question
            cpt_code: CPT code (optional)
            context: Additional context (optional)
        
        Returns:
            Final state with structured_answer
        """
        # Initialize state
        initial_state: AgenticRAGState = {
            "question": question,
            "cpt_code": cpt_code,
            "context": context,
            "question_type": None,
            "retrieval_strategy": None,
            "query_candidates": [],
            "retrieved_chunks": [],
            "retrieval_metadata": {},
            "evidence_assessment": None,
            "retry_count": 0,
            "refined_queries": [],
            "structured_answer": None,
            "messages": [],
            "error": None
        }
        
        # Run workflow
        final_state = self.graph.invoke(initial_state)
        
        return final_state
    
    def stream(self, question: str, cpt_code: int = None, context: str = None):
        """
        Stream the workflow execution (for debugging/monitoring)
        
        Yields:
            (node_name, state) tuples
        """
        initial_state: AgenticRAGState = {
            "question": question,
            "cpt_code": cpt_code,
            "extracted_cpt_codes": [],
            "max_retry_allowed": 2,
            "question_complexity": None,
            "context": context,
            "question_type": None,
            "retrieval_strategy": None,
            "query_candidates": [],
            "retrieved_chunks": [],
            "retrieval_metadata": {},
            "evidence_assessment": None,
            "retry_count": 0,
            "refined_queries": [],
            "structured_answer": None,
            "messages": [],
            "error": None
        }
        
        for step in self.graph.stream(initial_state):
            yield step
