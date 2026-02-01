"""
Agent nodes for the Agentic RAG workflow
Each agent is a function that takes state and returns updated state
"""
import json
from typing import Dict, Any, List, TYPE_CHECKING
from openai import AzureOpenAI

from .state import (
    AgenticRAGState, 
    QueryCandidate, 
    EvidenceAssessment, 
    StructuredAnswer
)
from .config import AgenticRAGConfig

if TYPE_CHECKING:
    from .tools.retrieval_tools import RetrievalTools


class AgenticRAGAgents:
    """Collection of agent nodes for LangGraph workflow"""
    
    def __init__(self, config: AgenticRAGConfig):
        self.config = config
        self.llm_client = AzureOpenAI(
            api_key=config.azure_openai_api_key,
            api_version=config.azure_api_version,
            azure_endpoint=config.azure_openai_endpoint,
        )
    
    def _call_llm(self, system_prompt: str, user_prompt: str, response_format: str = "text") -> str:
        """Helper to call Azure OpenAI LLM"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.llm_client.chat.completions.create(
            model=self.config.azure_deployment_name,
            messages=messages,
            temperature=self.config.agent_temperature,
            max_tokens=self.config.agent_max_tokens,
        )
        
        return response.choices[0].message.content
    
    def orchestrator_node(self, state: AgenticRAGState) -> Dict[str, Any]:
        """
        🧠 Orchestrator Agent (Root)
        
        Responsibilities:
        1. Parse user intent & identify question type
        2. Extract CPT codes from question
        3. Select optimal retrieval strategy (BM25/Semantic/Hybrid)
        4. Control retry logic and decide max iterations
        
        Returns:
            question_type, retrieval_strategy, extracted_cpt_codes, max_retry_allowed
        """
        question = state["question"]
        cpt_code = state.get("cpt_code")
        context = state.get("context", "")
        retry_count = state.get("retry_count", 0)
        
        system_prompt = """You are the root orchestrator for NCCI medical coding policy analysis.

TASK: Analyze the user's question and make strategic decisions.

1. QUESTION TYPE Classification:
   - "modifier": Questions about allowed/not allowed modifiers for CPT codes
   - "PTP": Procedure-to-Procedure edit questions (billing two codes together)
   - "guideline": General coding rules and policies
   - "definition": What does this code/term mean?
   - "general": Other questions

2. RETRIEVAL STRATEGY Selection:
   - "bm25": Use for exact keyword match (CPT codes, specific modifiers, section names)
   - "semantic": Use for conceptual questions (definitions, explanations)
   - "hybrid": Use for complex questions needing both exact match and semantic understanding
   
3. CPT CODE Extraction:
   - Extract all CPT codes mentioned in the question
   - Example: "31622 and 31623" → ["31622", "31623"]

4. RETRY DECISION:
   - Consider question complexity
   - Simple questions (single CPT + modifier): max 1 retry
   - Complex questions (multiple CPTs, PTP edits): max 2 retries
   - Very complex (guideline interpretations): max 3 retries

Return JSON:
{
    "question_type": "modifier|PTP|guideline|definition|general",
    "retrieval_strategy": "bm25|semantic|hybrid",
    "extracted_cpt_codes": ["code1", "code2", ...],
    "max_retry_allowed": 1-3,
    "question_complexity": "simple|medium|complex",
    "reasoning": "brief explanation of decisions"
}"""
        
        user_prompt = f"""Question: {question}
CPT Code (provided): {cpt_code if cpt_code else "Not specified"}
Additional Context: {context if context else "None"}
Current Retry Count: {retry_count}

Analyze this question and make strategic decisions."""
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            # Parse JSON response
            result = json.loads(response)
            
            # Update state with orchestrator decisions
            return {
                "question_type": result.get("question_type", "general"),
                "retrieval_strategy": result.get("retrieval_strategy", "hybrid"),
                "extracted_cpt_codes": result.get("extracted_cpt_codes", []),
                "max_retry_allowed": result.get("max_retry_allowed", 2),
                "question_complexity": result.get("question_complexity", "medium"),
                "messages": state.get("messages", []) + [
                    f"Orchestrator: Type={result.get('question_type')}, "
                    f"Strategy={result.get('retrieval_strategy')}, "
                    f"CPTs={result.get('extracted_cpt_codes')}, "
                    f"MaxRetry={result.get('max_retry_allowed')}"
                ]
            }
        except Exception as e:
            # Fallback to safe defaults
            return {
                "question_type": "general",
                "retrieval_strategy": "hybrid",
                "extracted_cpt_codes": [str(cpt_code)] if cpt_code else [],
                "max_retry_allowed": 2,
                "question_complexity": "medium",
                "messages": state.get("messages", []) + [
                    f"Orchestrator: Error - {str(e)}, using safe defaults"
                ],
                "error": str(e)
            }
    
    def query_planner_node(self, state: AgenticRAGState) -> Dict[str, Any]:
        """
        🧭 Query Planner Agent
        
        Responsibilities:
        1. Generate multiple query candidates for retrieval
        2. Include variations: original, expanded, synonym, section-specific
        3. Assign weights to different query types
        4. Consider question type for targeted query generation
        
        Returns:
            query_candidates: List of QueryCandidate objects
        """
        question = state["question"]
        cpt_code = state.get("cpt_code")
        question_type = state.get("question_type", "general")
        extracted_cpt_codes = state.get("extracted_cpt_codes", [])
        
        # Use extracted CPT codes if available
        if extracted_cpt_codes and not cpt_code:
            cpt_code = extracted_cpt_codes[0] if len(extracted_cpt_codes) == 1 else None
        
        system_prompt = """You are an expert query generation specialist for NCCI medical coding documentation.

TASK: Generate 3-5 diverse query candidates to maximize retrieval coverage.

QUERY TYPES:
1. "original": Clean version of user's question
2. "expanded": Add medical coding terminology and context
3. "synonym": Use abbreviations and alternative terms (e.g., "PTP" → "Procedure-to-Procedure Edit")
4. "section_specific": Target specific sections (e.g., "Modifier Section", "PTP Edit Tables")

WEIGHT GUIDELINES:
- original: 1.0 (baseline)
- expanded: 1.2 (usually more informative)
- synonym: 0.9 (catch alternative phrasings)
- section_specific: 1.1 (focused retrieval)

QUESTION TYPE STRATEGIES:
- "modifier": Include "modifier", "append", CPT code, "allowed/not allowed"
- "PTP": Include both CPT codes, "PTP edit", "column 1", "column 2", "modifier indicator"
- "guideline": Include policy keywords, "rule", "guideline", "requirement"
- "definition": Include term, "definition", "description", "means"

Return JSON array:
[
    {"query": "exact query text", "query_type": "original|expanded|synonym|section_specific", "weight": float},
    ...
]

Generate 3-5 candidates."""
        
        user_prompt = f"""Question: {question}
CPT Code: {cpt_code if cpt_code else "Not specified"}
All CPT Codes: {extracted_cpt_codes if extracted_cpt_codes else "None"}
Question Type: {question_type}

Generate diverse query candidates optimized for {question_type} questions."""
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            queries_data = json.loads(response)
            
            # Validate and create QueryCandidate objects
            query_candidates = []
            for q in queries_data:
                try:
                    query_candidates.append(QueryCandidate(**q))
                except Exception:
                    # Skip invalid candidates
                    continue
            
            # Ensure at least one query
            if not query_candidates:
                query_candidates = [QueryCandidate(
                    query=question,
                    query_type="original",
                    weight=1.0
                )]
            
            return {
                "query_candidates": query_candidates,
                "messages": state.get("messages", []) + [
                    f"Query Planner: Generated {len(query_candidates)} candidates: " +
                    ", ".join([f"{q.query_type}({q.weight})" for q in query_candidates])
                ]
            }
        except Exception as e:
            # Fallback: use original question
            fallback = [QueryCandidate(
                query=question,
                query_type="original",
                weight=1.0
            )]
            return {
                "query_candidates": fallback,
                "messages": state.get("messages", []) + [
                    f"Query Planner: Error - {str(e)}, using original question"
                ],
                "error": str(e)
            }
    
    def evidence_judge_node(self, state: AgenticRAGState) -> Dict[str, Any]:
        """
        🧪 Evidence Judge Agent (Quality Gate)
        
        Responsibilities:
        1. Assess evidence coverage - does it address all question aspects?
        2. Assess evidence specificity - is it relevant to the CPT codes?
        3. Check for contradictions or conflicts
        4. Count valid citations
        5. Identify missing aspects for refinement
        6. DECIDE: sufficient → extract | insufficient → refine & retry
        
        Decision Thresholds:
        - SUFFICIENT if: coverage ≥ 0.7 AND specificity ≥ 0.6 AND no major contradictions
        - RETRY if: coverage < 0.7 OR specificity < 0.5 OR missing critical aspects
        
        Returns:
            evidence_assessment: EvidenceAssessment object with decision
        """
        question = state["question"]
        question_type = state.get("question_type", "general")
        extracted_cpt_codes = state.get("extracted_cpt_codes", [])
        retrieved_chunks = state.get("retrieved_chunks", [])
        retry_count = state.get("retry_count", 0)
        
        # No evidence retrieved
        if not retrieved_chunks:
            return {
                "evidence_assessment": EvidenceAssessment(
                    is_sufficient=False,
                    coverage_score=0.0,
                    specificity_score=0.0,
                    citation_count=0,
                    has_contradiction=False,
                    reasoning="No chunks retrieved - need to retry with different queries",
                    missing_aspects=["all aspects - no evidence found"]
                ),
                "messages": state.get("messages", []) + [
                    "Evidence Judge: ❌ No evidence found - retry needed"
                ]
            }
        
        # Prepare evidence text (limit to first 10 chunks for LLM)
        evidence_text = "\n\n".join([
            f"[Chunk {i+1}] (ID: {chunk.chunk_id}, Score: {chunk.score:.3f})\n{chunk.text[:600]}..."
            for i, chunk in enumerate(retrieved_chunks[:10])
        ])
        
        system_prompt = """You are an expert evidence quality judge for NCCI medical coding documentation.

TASK: Evaluate if retrieved evidence is SUFFICIENT to answer the user's question.

EVALUATION CRITERIA:

1. COVERAGE SCORE (0.0-1.0):
   - 1.0: Evidence fully addresses ALL aspects of the question
   - 0.7-0.9: Evidence covers most aspects, minor gaps
   - 0.4-0.6: Evidence covers some aspects, significant gaps
   - 0.0-0.3: Evidence barely relevant or missing key information
   
   For modifier questions: Must have modifier list for specific CPT
   For PTP questions: Must have both CPT codes and edit indicator
   For guideline questions: Must have relevant policy statements

2. SPECIFICITY SCORE (0.0-1.0):
   - 1.0: Evidence is exactly about the CPT codes/modifiers asked
   - 0.7-0.9: Evidence is relevant but includes extra info
   - 0.4-0.6: Evidence is general, not specific to the question
   - 0.0-0.3: Evidence is too general or off-topic

3. CITATION COUNT:
   - How many chunks directly support answering the question?
   - Minimum 2 citations for simple questions, 3+ for complex

4. CONTRADICTIONS:
   - Are there conflicting statements in different chunks?
   - Do chunks from different pages contradict each other?

5. MISSING ASPECTS:
   - What specific information is missing?
   - Be concrete: "missing modifier 59 status", "missing column 2 codes"

DECISION RULE:
- is_sufficient = TRUE if: coverage ≥ 0.7 AND specificity ≥ 0.6 AND citation_count ≥ 2 AND !has_contradiction
- is_sufficient = FALSE otherwise → trigger retry

Return JSON:
{
    "is_sufficient": true/false,
    "coverage_score": 0.0-1.0,
    "specificity_score": 0.0-1.0,
    "citation_count": integer,
    "has_contradiction": true/false,
    "reasoning": "detailed explanation of the decision",
    "missing_aspects": ["specific missing item 1", "specific missing item 2", ...]
}"""
        
        user_prompt = f"""Question: {question}
Question Type: {question_type}
CPT Codes in Question: {extracted_cpt_codes if extracted_cpt_codes else "None"}
Number of Retrieved Chunks: {len(retrieved_chunks)}
Current Retry Count: {retry_count}

Retrieved Evidence:
{evidence_text}

Evaluate the evidence quality and decide if it's sufficient to answer the question."""
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            assessment_data = json.loads(response)
            assessment = EvidenceAssessment(**assessment_data)
            
            # Log decision
            decision_emoji = "✅" if assessment.is_sufficient else "❌"
            return {
                "evidence_assessment": assessment,
                "messages": state.get("messages", []) + [
                    f"Evidence Judge: {decision_emoji} Sufficient={assessment.is_sufficient}, "
                    f"Coverage={assessment.coverage_score:.2f}, "
                    f"Specificity={assessment.specificity_score:.2f}, "
                    f"Citations={assessment.citation_count}, "
                    f"Contradictions={assessment.has_contradiction}"
                ]
            }
        except Exception as e:
            # Fallback: conservative assessment
            # If we have enough chunks and this is first try, be optimistic
            # If this is a retry, be more lenient to avoid infinite loop
            is_sufficient_fallback = (
                len(retrieved_chunks) >= self.config.min_citation_count and
                (retry_count == 0 or retry_count >= 2)
            )
            
            fallback = EvidenceAssessment(
                is_sufficient=is_sufficient_fallback,
                coverage_score=0.6,
                specificity_score=0.6,
                citation_count=len(retrieved_chunks),
                has_contradiction=False,
                reasoning=f"Error in assessment: {str(e)}. Fallback decision based on chunk count.",
                missing_aspects=[] if is_sufficient_fallback else ["Unable to assess - LLM error"]
            )
            return {
                "evidence_assessment": fallback,
                "messages": state.get("messages", []) + [
                    f"Evidence Judge: ⚠️ Error - {str(e)}, using fallback (sufficient={is_sufficient_fallback})"
                ],
                "error": str(e)
            }
    
    def query_refiner_node(self, state: AgenticRAGState) -> Dict[str, Any]:
        """
        🔁 Query Refiner Agent (Retry Loop)
        
        Responsibilities:
        1. Analyze missing aspects from Evidence Judge
        2. Generate targeted refined queries to fill gaps
        3. Consider previous failed queries to avoid repetition
        4. Provide specific section hints or keyword expansions
        
        Strategy:
        - If missing specific CPT info → add CPT code explicitly
        - If missing modifier details → add "modifier" + section keywords
        - If missing PTP edit → add "PTP edit table" + both CPT codes
        - If too general → add specific section names
        
        Returns:
            refined_queries: List of string queries (not QueryCandidate objects)
        """
        question = state["question"]
        question_type = state.get("question_type", "general")
        assessment = state.get("evidence_assessment")
        query_candidates = state.get("query_candidates", [])
        retry_count = state.get("retry_count", 0)
        extracted_cpt_codes = state.get("extracted_cpt_codes", [])
        
        # If no missing aspects, no refinement needed
        if not assessment or not assessment.missing_aspects:
            return {
                "refined_queries": [],
                "messages": state.get("messages", []) + [
                    "Query Refiner: No missing aspects - no refinement needed"
                ]
            }
        
        # Extract previous queries to avoid repetition
        previous_queries = [qc.query for qc in query_candidates]
        
        system_prompt = """You are an expert query refinement specialist for NCCI medical coding documentation.

TASK: Generate 2-3 targeted refined queries to find MISSING information.

REFINEMENT STRATEGIES:

1. ADD SPECIFICITY:
   - If missing CPT-specific info → add CPT code explicitly
   - If missing modifier info → add "modifier" + CPT + "allowed/not allowed"
   - If missing section → add section name: "Modifier Section", "PTP Edit", "General Guidelines"

2. EXPAND KEYWORDS:
   - "modifier" → "modifier", "append", "addition code"
   - "PTP" → "PTP edit", "Procedure-to-Procedure", "Column 1", "Column 2"
   - "allowed" → "allowed", "appropriate", "may be reported with"

3. TARGET SECTIONS:
   - For modifiers: "CPT [code] modifier section"
   - For PTP: "CPT [code1] [code2] PTP edit table"
   - For guidelines: "General policy for [topic]"

4. AVOID REPETITION:
   - DO NOT generate queries similar to previous ones
   - If previous query was too broad, be more specific
   - If previous query was too narrow, broaden slightly

IMPORTANT:
- Return ONLY the query strings, NOT objects
- Each query should target a SPECIFIC missing aspect
- Queries should be different from previous attempts

Return JSON array of strings:
["refined query 1 targeting missing aspect X", "refined query 2 targeting missing aspect Y", ...]"""
        
        user_prompt = f"""Original Question: {question}
Question Type: {question_type}
CPT Codes: {extracted_cpt_codes if extracted_cpt_codes else "None"}
Retry Count: {retry_count}

Missing Aspects (from Evidence Judge):
{chr(10).join(f'- {aspect}' for aspect in assessment.missing_aspects)}

Previous Queries (AVOID similar):
{chr(10).join(f'- {q}' for q in previous_queries)}

Coverage Score: {assessment.coverage_score:.2f}
Specificity Score: {assessment.specificity_score:.2f}

Generate 2-3 refined queries to find the missing information."""
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            refined_queries = json.loads(response)
            
            # Ensure it's a list of strings
            if not isinstance(refined_queries, list):
                refined_queries = [str(refined_queries)]
            
            # Limit to 3 queries max
            refined_queries = refined_queries[:3]
            
            return {
                "refined_queries": refined_queries,
                "messages": state.get("messages", []) + [
                    f"Query Refiner: 🔁 Generated {len(refined_queries)} refined queries: " +
                    ", ".join([f'"{q[:50]}..."' for q in refined_queries])
                ]
            }
        except Exception as e:
            # Fallback: create simple refined queries based on missing aspects
            fallback_queries = []
            for aspect in assessment.missing_aspects[:2]:
                # Simple concatenation of question + missing aspect
                fallback_queries.append(f"{question} {aspect}")
            
            if not fallback_queries:
                fallback_queries = [question]  # Ultimate fallback
            
            return {
                "refined_queries": fallback_queries,
                "messages": state.get("messages", []) + [
                    f"Query Refiner: ⚠️ Error - {str(e)}, using fallback queries"
                ],
                "error": str(e)
            }
    
    def structured_extraction_node(self, state: AgenticRAGState) -> Dict[str, Any]:
        """
        📦 Structured Extraction Agent
        - Extract structured answer with evidence
        """
        question = state["question"]
        retrieved_chunks = state.get("retrieved_chunks", [])
        
        # Prepare evidence text
        evidence_text = "\n\n".join([
            f"[{i+1}] {chunk.text}\n(chunk_id: {chunk.chunk_id}, metadata: {chunk.metadata})"
            for i, chunk in enumerate(retrieved_chunks)
        ])
        
        system_prompt = """You are an expert NCCI medical coding policy analyst.
Extract structured information from the evidence to answer the question.

Return JSON format:
{
    "answer": "comprehensive answer",
    "rules": ["rule 1", "rule 2", ...],
    "allowed_modifiers": ["59", "XE", ...],
    "constraints": ["constraint 1", ...],
    "risks": ["risk 1", ...],
    "evidence_trace": [
        {"chunk_id": "...", "quote": "...", "section": "...", "page": "..."},
        ...
    ],
    "confidence": 0.0-1.0
}"""
        
        user_prompt = f"""Question: {question}

Evidence:
{evidence_text}

Extract structured answer with full evidence traceability."""
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            answer_data = json.loads(response)
            structured_answer = StructuredAnswer(**answer_data)
            
            return {
                "structured_answer": structured_answer,
                "messages": state.get("messages", []) + [
                    f"Structured Extraction: Generated answer with confidence {structured_answer.confidence:.2f}"
                ]
            }
        except Exception as e:
            # Fallback: basic answer
            fallback = StructuredAnswer(
                answer=f"Error in extraction: {str(e)}. Retrieved {len(retrieved_chunks)} chunks.",
                confidence=0.0
            )
            return {
                "structured_answer": fallback,
                "messages": state.get("messages", []) + [
                    f"Structured Extraction: Error - {str(e)}"
                ],
                "error": str(e)
            }
    
    def retrieval_router_node(self, state: AgenticRAGState, tools: 'RetrievalTools') -> Dict[str, Any]:
        """
        🔧 Retrieval Tool Router Agent
        
        Responsibilities:
        1. Determine which retrieval tools to use based on strategy
        2. Execute retrieval with appropriate query candidates
        3. Handle both initial retrieval and retry retrieval
        4. Merge and deduplicate results
        5. Return top-k chunks
        
        Tool Selection:
        - bm25: Use BM25 tool only (keyword match)
        - semantic: Use Semantic tool only (vector similarity)
        - hybrid: Use multi_query_hybrid_search (BM25 + Semantic fusion with RRF)
        
        Args:
            state: Current workflow state
            tools: RetrievalTools instance
        
        Returns:
            retrieved_chunks, retrieval_metadata
        """
        strategy = state.get("retrieval_strategy", "hybrid")
        query_candidates = state.get("query_candidates", [])
        refined_queries = state.get("refined_queries", [])
        cpt_code = state.get("cpt_code")
        extracted_cpt_codes = state.get("extracted_cpt_codes", [])
        retry_count = state.get("retry_count", 0)
        
        # Determine top_k based on retry count (get more on retries)
        top_k = self.config.top_k * (1 + retry_count)  # Increase top_k on retries
        
        # Prepare queries
        # If in retry mode, use refined queries
        if refined_queries and retry_count > 0:
            query_candidates = [
                QueryCandidate(
                    query=q,
                    query_type="refined",
                    weight=1.3  # Boost refined queries
                )
                for q in refined_queries
            ]
            query_source = "refined"
        elif not query_candidates:
            # Fallback to original question
            query_candidates = [
                QueryCandidate(
                    query=state["question"],
                    query_type="original",
                    weight=1.0
                )
            ]
            query_source = "fallback"
        else:
            query_source = "planner"
        
        # Determine CPT code for range-based boosting
        if not cpt_code and extracted_cpt_codes:
            cpt_code = extracted_cpt_codes[0] if len(extracted_cpt_codes) == 1 else None
        
        # Execute retrieval based on strategy
        retrieved_chunks = []
        metadata = {}
        
        try:
            if strategy == "bm25":
                # BM25 only
                results = []
                for candidate in query_candidates:
                    bm25_results = tools.bm25_search(candidate.query, top_k=top_k)
                    # Apply candidate weight
                    for r in bm25_results:
                        r.score *= candidate.weight
                    results.extend(bm25_results)
                
                # Deduplicate and re-rank
                seen = set()
                unique_results = []
                for r in sorted(results, key=lambda x: x.score, reverse=True):
                    if r.chunk_id not in seen:
                        seen.add(r.chunk_id)
                        unique_results.append(r)
                
                retrieved_chunks = unique_results[:self.config.top_k]
                metadata = {
                    "strategy": "bm25",
                    "count": len(retrieved_chunks),
                    "query_source": query_source,
                    "num_queries": len(query_candidates)
                }
                
            elif strategy == "semantic":
                # Semantic only
                results = []
                for candidate in query_candidates:
                    semantic_results = tools.semantic_search(candidate.query, top_k=top_k)
                    # Apply candidate weight
                    for r in semantic_results:
                        r.score *= candidate.weight
                    results.extend(semantic_results)
                
                # Deduplicate and re-rank
                seen = set()
                unique_results = []
                for r in sorted(results, key=lambda x: x.score, reverse=True):
                    if r.chunk_id not in seen:
                        seen.add(r.chunk_id)
                        unique_results.append(r)
                
                retrieved_chunks = unique_results[:self.config.top_k]
                metadata = {
                    "strategy": "semantic",
                    "count": len(retrieved_chunks),
                    "query_source": query_source,
                    "num_queries": len(query_candidates)
                }
                
            else:
                # Hybrid (default) - use multi_query_hybrid_search
                retrieved_chunks, metadata = tools.multi_query_hybrid_search(
                    query_candidates,
                    cpt_code=cpt_code,
                    top_k=self.config.top_k
                )
                metadata["query_source"] = query_source
            
            return {
                "retrieved_chunks": retrieved_chunks,
                "retrieval_metadata": metadata,
                "messages": state.get("messages", []) + [
                    f"Retrieval Router: {strategy.upper()} retrieved {len(retrieved_chunks)} chunks "
                    f"from {len(query_candidates)} queries (source: {query_source})"
                ]
            }
            
        except Exception as e:
            # Fallback: empty results
            return {
                "retrieved_chunks": [],
                "retrieval_metadata": {"error": str(e), "strategy": strategy},
                "messages": state.get("messages", []) + [
                    f"Retrieval Router: ❌ Error - {str(e)}"
                ],
                "error": str(e)
            }
