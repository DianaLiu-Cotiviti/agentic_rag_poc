"""
Prompt templates for Agentic RAG system.
Centralized management for easy iteration and A/B testing.
"""

from .orchestrator_prompts import build_orchestrator_prompt, ORCHESTRATOR_SYSTEM_MESSAGE
from .query_planner_prompts import build_query_planner_prompt, QUERY_PLANNER_SYSTEM_MESSAGE
from .retrieval_router_prompts import build_retrieval_router_prompt, RETRIEVAL_ROUTER_SYSTEM_MESSAGE
from .evidence_judge_prompts import EVIDENCE_JUDGE_PROMPTS
from .extraction_prompts import EXTRACTION_PROMPTS

__all__ = [
    "ORCHESTRATOR_PROMPTS",
    "QUERY_PLANNER_PROMPTS", 
    "EVIDENCE_JUDGE_PROMPTS",
    "EXTRACTION_PROMPTS"
]
