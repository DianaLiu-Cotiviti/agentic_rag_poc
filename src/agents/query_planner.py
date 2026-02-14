"""
Query Planner Agent - Query Decomposition and Structuring
Responsible for:
- Generating multiple search-optimized sub-queries
- Extracting user intents and entities
- Identifying constraints and providing retrieval hints
"""

from pydantic import BaseModel, Field
from typing import Literal, List, Dict
from .base import BaseAgent
from ..state import AgenticRAGState, QueryCandidate, SearchGuidance
from ..prompts.query_planner_prompts import build_query_planner_prompt, QUERY_PLANNER_SYSTEM_MESSAGE
from ..prompts.search_guidance_templates import select_guidance_for_query


class QueryCandidateOutput(BaseModel):
    """Query candidate from LLM output (without guidance)"""
    query: str
    query_type: Literal["original", "expanded", "synonym", "section_specific", "constraint_focused"]
    weight: float = 1.0


class EntityExtraction(BaseModel):
    """Structured entity extraction from query"""
    cpt_codes: List[str] = Field(default_factory=list, description="CPT codes mentioned in query")
    modifiers: List[str] = Field(default_factory=list, description="Modifier numbers (e.g., '59', '25')")
    procedures: List[str] = Field(default_factory=list, description="Medical procedure names")
    anatomical_terms: List[str] = Field(default_factory=list, description="Body parts or anatomical regions")
    policy_terms: List[str] = Field(default_factory=list, description="Billing/coding policy terms")
    conditions: List[str] = Field(default_factory=list, description="Temporal, spatial, or conditional constraints")


class QueryPlannerDecision(BaseModel):
    """
    Query Planner Decision Schema
    
    Structures the user query into sub-queries, intents, entities, and retrieval hints
    """
    # Query Candidates (Sub-queries)
    query_candidates: List[QueryCandidateOutput] = Field(
        description="2-4 search-optimized query variants",
        min_length=2,
        max_length=4
    )
    
    # Intent Analysis
    primary_intent: Literal["lookup", "validation", "comparison", "procedural", "constraint_check", "policy_inquiry"] = Field(
        description="Primary user intent/goal"
    )
    secondary_intents: List[str] = Field(
        default_factory=list,
        description="Additional underlying intents (optional)",
        max_length=3
    )
    
    # Entity Extraction
    entities: EntityExtraction = Field(
        description="Structured entities extracted from query"
    )
    
    # Constraints
    constraints: List[str] = Field(
        default_factory=list,
        description="Query constraints (temporal, spatial, conditional, scope)",
        max_length=5
    )
    
    # Retrieval Hints
    retrieval_hints: List[str] = Field(
        description="Actionable hints for retrieval system optimization",
        min_length=1,
        max_length=8
    )
    
    # Reasoning
    reasoning: str = Field(
        description="Brief explanation of query planning decisions (2-3 sentences)",
        min_length=50,
        max_length=500
    )


class QueryPlannerAgent(BaseAgent):
    """
    Query Planner Agent - Query Decomposition Expert
    
    Responsibilities:
    1. Generate 2-4 search-optimized query candidates (sub-queries)
    2. Extract primary and secondary intents
    3. Extract structured entities (CPT codes, modifiers, procedures, etc.)
    4. Identify query constraints
    5. Provide retrieval hints for optimization
    """
    
    def process(self, state: AgenticRAGState) -> dict:
        """
        Structure and decompose user query
        
        Args:
            state: Contains question, question_type, question_keywords
            
        Returns:
            dict: Query planning decisions including query_candidates, intents, entities, constraints, hints
        """
        question = state["question"]
        question_type = state.get("question_type", "general")
        keywords = state.get("question_keywords", [])
        
        # Build prompt from prompt module
        prompt = build_query_planner_prompt(question, question_type, keywords)
        
        # Call LLM with structured output
        response = self.client.beta.chat.completions.parse(
            model=self.config.azure_deployment_name,
            messages=[
                {"role": "system", "content": QUERY_PLANNER_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            response_format=QueryPlannerDecision,
            temperature=self.config.agent_temperature,
            max_tokens=1500  # Limit query planning output
        )
        
        decision = response.choices[0].message.parsed
        
        # Extract entities for guidance generation
        cpt_codes = decision.entities.cpt_codes if decision.entities else []
        modifiers = decision.entities.modifiers if decision.entities else []
        
        # Generate query candidates with search guidance
        query_candidates_with_guidance = []
        for qc in decision.query_candidates:
            # Automatically generate appropriate guidance for each query
            # Use semantic guidance by default (retrieval tools will adapt for hybrid)
            guidance_text = select_guidance_for_query(
                query_type=qc.query_type,
                question_type=question_type,
                cpt_codes=cpt_codes,
                modifiers=modifiers,
                retrieval_method="semantic"
            )
            
            # Create SearchGuidance object
            guidance = SearchGuidance(
                semantic_guidance=guidance_text,
                boost_terms=cpt_codes + modifiers,  # Basic boost terms
                target_doc_types=[],  # Will be filled by retrieval tools
                expected_sections=[],  # Will be filled by retrieval tools
                metadata_filters={}  # Will be filled by retrieval tools
            )
            
            # Create QueryCandidate with guidance
            candidate = QueryCandidate(
                query=qc.query,
                query_type=qc.query_type,
                weight=qc.weight,
                guidance=guidance
            )
            query_candidates_with_guidance.append(candidate)
        
        # Convert to state-compatible format
        return {
            "query_candidates": query_candidates_with_guidance,
            "retrieval_hints": decision.retrieval_hints if decision.retrieval_hints else [],
            # Store additional planning metadata in messages for debugging
            # "messages": [
            #     f"Primary Intent: {decision.primary_intent}",
            #     f"Secondary Intents: {', '.join(decision.secondary_intents) if decision.secondary_intents else 'None'}",
            #     f"Entities - CPT Codes: {', '.join(decision.entities.cpt_codes) if decision.entities.cpt_codes else 'None'}",
            #     f"Entities - Modifiers: {', '.join(decision.entities.modifiers) if decision.entities.modifiers else 'None'}",
            #     f"Entities - Procedures: {', '.join(decision.entities.procedures) if decision.entities.procedures else 'None'}",
            #     f"Constraints: {', '.join(decision.constraints) if decision.constraints else 'None'}",
            #     f"Retrieval Hints: {'; '.join(decision.retrieval_hints[:3])}...",  # First 3 hints
            #     f"Query Planner Reasoning: {decision.reasoning}"
            # ]
        }
