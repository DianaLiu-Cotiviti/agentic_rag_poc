"""
Orchestrator Agent - Global Controller
Responsible for macro-level strategy decisions:
- Query analysis and keyword extraction
- Retrieval strategy selection (range routing, RAG, hybrid)
- Retry control and iteration strategy
- Structured extraction requirements
"""

from pydantic import BaseModel, Field
from typing import Literal, List
from .base import BaseAgent
from ..state import AgenticRAGState
from ..prompts.orchestrator_prompts import build_orchestrator_prompt, ORCHESTRATOR_SYSTEM_MESSAGE


class OrchestratorDecision(BaseModel):
    """
    Orchestrator Global Decision Schema
    
    This schema defines the macro-level control decisions for the entire RAG pipeline.
    """
    # Question Analysis
    question_type: Literal["modifier", "PTP", "guideline", "definition", "comparison", "procedural", "general"] = Field(
        description="Type of medical coding question"
    )
    question_keywords: List[str] = Field(
        description="Key concepts, CPT codes, and medical terms extracted from the query",
        min_items=1,
        max_items=10
    )
    question_complexity: Literal["simple", "medium", "complex"] = Field(
        description="Complexity level affecting retrieval and processing strategies"
    )
    
    # Retrieval Strategy Hints (suggestions for Retrieval Router)
    retrieval_strategies: List[Literal["range_routing", "bm25", "semantic", "hybrid"]] = Field(
        description=(
            "Strategy hints for Retrieval Router. Ordered list of suggested retrieval methods:\n"
            "- range_routing: Section/CPT-based pre-filtering (use when query has CPT codes)\n"
            "- bm25: Keyword-based retrieval (use for exact terms, codes, modifiers)\n"
            "- semantic: Embedding-based retrieval (use for conceptual queries)\n"
            "- hybrid: Combined BM25+Semantic (use for mixed keyword+conceptual queries)\n"
            "Note: These are HINTS - Retrieval Router may adapt based on mode (direct/planning/tool_calling)"
        ),
        min_items=1,
        max_items=3
    )
    
    # Iteration Control
    enable_retry: bool = Field(
        description="Whether to enable iterative retrieval if initial evidence is insufficient"
    )
    max_retry_allowed: int = Field(
        ge=0,
        le=3,
        description="Maximum number of retry iterations allowed (0 means no retry)"
    )
    
    # Output Requirements
    require_structured_output: bool = Field(
        description="Whether the final answer requires structured extraction (rules, modifiers, constraints)"
    )
    
    # Reasoning
    reasoning: str = Field(
        description="Brief explanation of the strategy decisions (2-3 sentences)",
        min_length=20,
        max_length=500
    )


class OrchestratorAgent(BaseAgent):
    """
    Orchestrator Agent - Global Strategy Controller
    
    Responsibilities:
    1. Analyze question and extract key concepts/terms
    2. Determine optimal retrieval strategy pipeline
    3. Decide on iteration and retry mechanisms
    4. Specify output structure requirements
    5. Provide macro-level routing decisions
    """
    
    def process(self, state: AgenticRAGState) -> dict:
        """
        Analyze question and generate global strategy decisions
        
        Args:
            state: Contains question and optional context
            
        Returns:
            dict: Global strategy decisions including retrieval_strategies, 
                  enable_retry, require_structured_output, etc.
        """
        question = state["question"]
        context = state.get("context", "")
        
        # Build prompt from prompt module
        prompt = build_orchestrator_prompt(question, context)
        
        # Call LLM with structured output
        response = self.client.beta.chat.completions.parse(
            model=self.config.azure_deployment_name,
            messages=[
                {"role": "system", "content": ORCHESTRATOR_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            response_format=OrchestratorDecision,
            temperature=self.config.agent_temperature,
            max_tokens=800  # Limit orchestrator decision output
        )
        
        decision = response.choices[0].message.parsed
        
        # Return strategy decisions (mode is determined by config, not orchestrator)
        return {
            "question_type": decision.question_type,
            "question_keywords": decision.question_keywords,
            "question_complexity": decision.question_complexity,
            "retrieval_strategies": decision.retrieval_strategies,  # Strategy HINTS for router
            "enable_retry": decision.enable_retry,
            "max_retry_allowed": decision.max_retry_allowed,
            "require_structured_output": decision.require_structured_output,
            "orchestrator_reasoning": decision.reasoning
        }
