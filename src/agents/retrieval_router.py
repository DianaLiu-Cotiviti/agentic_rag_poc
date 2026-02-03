"""
Retrieval Router Agent - Retrieval Execution Planner with Tool Calling
Responsible for:
- Creating detailed retrieval execution plans
- Configuring retrieval parameters
- Mapping queries to strategies
- Orchestrating parallel retrieval execution
- LLM-driven tool calling for dynamic retrieval
"""

from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any
import json
from .base import BaseAgent
from ..state import AgenticRAGState, RetrievalResult
from ..prompts.retrieval_router_prompts import (
    build_retrieval_router_prompt,
    build_tool_calling_prompt,
    RETRIEVAL_ROUTER_SYSTEM_MESSAGE
)


class PreFilteringConfig(BaseModel):
    """Pre-retrieval filtering configuration"""
    apply_range_routing: bool = Field(description="Whether to apply range routing pre-filter")
    range_filter_cpt_codes: List[str] = Field(default_factory=list, description="CPT codes for range filtering")
    range_filter_limit: int = Field(default=300, ge=100, le=1000, description="Max chunks from range routing")


class QueryStrategyMapping(BaseModel):
    """Maps a query candidate to retrieval strategies"""
    query_index: int = Field(description="Index in query_candidates list")
    strategies: List[Literal["bm25", "semantic", "hybrid"]] = Field(
        description="Which retrieval strategies to use for this query"
    )
    reasoning: str = Field(description="Why these strategies for this query")


class RetrievalParameters(BaseModel):
    """Retrieval parameters for each strategy"""
    bm25: Dict[str, Any] = Field(default_factory=lambda: {"top_k": 20, "boost_exact_match": True})
    semantic: Dict[str, Any] = Field(default_factory=lambda: {"top_k": 20, "similarity_threshold": 0.0})
    hybrid: Dict[str, Any] = Field(default_factory=lambda: {"top_k": 20, "bm25_weight": 0.5, "semantic_weight": 0.5})


class FusionParameters(BaseModel):
    """Result fusion configuration"""
    rrf_k: int = Field(default=60, ge=10, le=100, description="RRF k parameter")
    use_query_weights: bool = Field(default=True, description="Whether to use query candidate weights")
    boost_range_results: float = Field(default=1.5, ge=1.0, le=2.0, description="Boost factor for range-routed chunks")


class ExecutionStep(BaseModel):
    """Single execution step"""
    stage: Literal["pre_filter", "retrieval", "fusion"]
    action: str
    parallel: bool = False


class ExecutionPlan(BaseModel):
    """Execution plan for retrieval"""
    mode: Literal["sequential", "parallel", "hybrid"]
    steps: List[ExecutionStep]


class RetrievalRouterDecision(BaseModel):
    """
    Retrieval Router Decision Schema
    
    Defines the complete execution plan for retrieval
    """
    pre_filtering: PreFilteringConfig
    query_strategy_mapping: List[QueryStrategyMapping] = Field(min_length=1)
    retrieval_parameters: RetrievalParameters
    fusion_strategy: Literal["rrf", "weighted_rrf", "cascade", "parallel_with_dedup"]
    fusion_parameters: FusionParameters
    execution_plan: ExecutionPlan
    reasoning: str = Field(min_length=50, max_length=500)


class RetrievalRouterAgent(BaseAgent):
    """
    Retrieval Router Agent - Intelligent Retrieval Executor
    
    Responsibilities:
    1. Analyze query characteristics and retrieval hints
    2. Create detailed retrieval execution plan
    3. Execute retrieval with optimal parameters
    4. Fuse results from multiple queries and strategies
    
    Note: This agent has two modes:
    - Planning mode: Uses LLM to create execution plan (optional, for complex scenarios)
    - Execution mode: Directly executes based on Orchestrator's strategies (default)
    """
    
    def __init__(self, config, tools=None, mode="tool_calling"):
        """
        Initialize retrieval router
        
        Args:
            config: Configuration object
            tools: RetrievalTools instance (optional, can be initialized later)
            mode: "tool_calling" (LLM calls tools), "planning" (LLM plans, agent executes), "direct" (no LLM)
        """
        super().__init__(config)
        self.tools = tools
        self.mode = mode  # "tool_calling" | "planning" | "direct"
        self.max_tool_iterations = 10  # Max iterations for tool calling mode
    
    def process(self, state: AgenticRAGState) -> dict:
        """
        Execute retrieval based on strategies and query candidates
        
        Args:
            state: Contains retrieval_strategies, query_candidates, question_keywords
            
        Returns:
            dict: Contains retrieved_chunks and retrieval_metadata
        """
        if self.mode == "tool_calling":
            return self._process_with_tool_calling(state)
        elif self.mode == "planning":
            return self._process_with_llm_planning(state)
        else:
            return self._process_direct_execution(state)
    
    def _process_with_tool_calling(self, state: AgenticRAGState) -> dict:
        """
        LLM-driven tool calling mode: LLM decides which tools to call and in what order
        """
        if self.tools is None:
            raise ValueError("RetrievalTools not initialized!")
        
        question = state["question"]
        question_type = state.get("question_type", "general")
        retrieval_strategies = state.get("retrieval_strategies", ["hybrid"])
        query_candidates = state.get("query_candidates", [])
        question_keywords = state.get("question_keywords", [])
        
        # Build tool calling prompt
        prompt = build_tool_calling_prompt(
            question=question,
            question_type=question_type,
            retrieval_strategies=retrieval_strategies,
            query_candidates=[
                {
                    "query": qc.get("query") if isinstance(qc, dict) else qc.query,
                    "query_type": qc.get("query_type") if isinstance(qc, dict) else qc.query_type,
                    "weight": qc.get("weight", 1.0) if isinstance(qc, dict) else qc.weight
                }
                for qc in query_candidates
            ],
            question_keywords=question_keywords
        )
        
        # Build tool definitions for OpenAI function calling
        tools = self._build_tool_definitions()
        
        # Initialize conversation
        messages = [
            {"role": "system", "content": RETRIEVAL_ROUTER_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt}
        ]
        
        # Storage for tool call results
        tool_results = {
            "range_routing": [],
            "bm25_search": [],
            "semantic_search": [],
            "hybrid_search": [],
            "rrf_fusion": []
        }
        
        # Multi-turn conversation with tool calling
        iteration = 0
        while iteration < self.max_tool_iterations:
            iteration += 1
            
            response = self.client.chat.completions.create(
                model=self.config.azure_deployment_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.config.agent_temperature
            )
            
            assistant_message = response.choices[0].message
            messages.append(assistant_message.model_dump())
            
            # Check if LLM wants to call tools
            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Execute tool
                    result = self._execute_tool(function_name, function_args, tool_results)
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
            else:
                # LLM finished tool calling
                break
        
        # Aggregate all results
        final_results, metadata = self._aggregate_tool_results(tool_results, state)
        
        return {
            "retrieved_chunks": final_results,
            "retrieval_metadata": metadata
        }
    
    def _build_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Build OpenAI function calling tool definitions
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "range_routing",
                    "description": "Pre-filter chunks by CPT code range. Use this when the query contains specific CPT codes to narrow down the search space.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cpt_code": {
                                "type": "integer",
                                "description": "5-digit CPT code (e.g., 14301, 27700)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of chunks to retrieve (default: 300)",
                                "default": 300
                            }
                        },
                        "required": ["cpt_code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "bm25_search",
                    "description": "Keyword-based search using BM25 algorithm. Best for exact term matching, specific codes, and precise phrases.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query text"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to retrieve (default: 20)",
                                "default": 20
                            },
                            "boost_exact_match": {
                                "type": "boolean",
                                "description": "Whether to boost exact code/modifier matches",
                                "default": True
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "semantic_search",
                    "description": "Vector-based semantic search. Best for conceptual queries, definitions, and understanding context.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query text"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to retrieve (default: 20)",
                                "default": 20
                            },
                            "similarity_threshold": {
                                "type": "number",
                                "description": "Minimum similarity score (0.0-1.0, default: 0.0)",
                                "default": 0.0
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "hybrid_search",
                    "description": "Combined BM25 + semantic search with weighted fusion. Best for balanced exact + conceptual matching.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query text"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to retrieve (default: 20)",
                                "default": 20
                            },
                            "bm25_weight": {
                                "type": "number",
                                "description": "Weight for BM25 scores (0.0-1.0, default: 0.5)",
                                "default": 0.5
                            },
                            "semantic_weight": {
                                "type": "number",
                                "description": "Weight for semantic scores (0.0-1.0, default: 0.5)",
                                "default": 0.5
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "rrf_fusion",
                    "description": "Fuse multiple result sets using Reciprocal Rank Fusion (RRF). Use after collecting results from multiple searches.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "result_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "IDs of previous search results to fuse (e.g., ['bm25_1', 'semantic_1'])"
                            },
                            "rrf_k": {
                                "type": "integer",
                                "description": "RRF k parameter (default: 60, higher = less aggressive fusion)",
                                "default": 60
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of final results after fusion (default: 20)",
                                "default": 20
                            }
                        },
                        "required": ["result_ids"]
                    }
                }
            }
        ]
    
    def _execute_tool(self, function_name: str, args: Dict[str, Any], tool_results: Dict) -> Dict:
        """
        Execute a retrieval tool and store results
        """
        try:
            if function_name == "range_routing":
                cpt_code = args["cpt_code"]
                limit = args.get("limit", 300)
                chunk_ids = self.tools.range_routing(cpt_code, limit=limit)
                
                result_id = f"range_{cpt_code}"
                tool_results["range_routing"].append({
                    "id": result_id,
                    "cpt_code": cpt_code,
                    "chunk_ids": list(chunk_ids),
                    "count": len(chunk_ids)
                })
                
                return {
                    "result_id": result_id,
                    "success": True,
                    "chunks_found": len(chunk_ids),
                    "message": f"Found {len(chunk_ids)} chunks in CPT {cpt_code} range"
                }
            
            elif function_name == "bm25_search":
                query = args["query"]
                top_k = args.get("top_k", 20)
                results = self.tools.bm25_search(query, top_k=top_k)
                
                result_id = f"bm25_{len(tool_results['bm25_search'])}"
                tool_results["bm25_search"].append({
                    "id": result_id,
                    "query": query,
                    "results": results
                })
                
                return {
                    "result_id": result_id,
                    "success": True,
                    "chunks_found": len(results),
                    "top_score": results[0].score if results else 0.0,
                    "message": f"BM25 search returned {len(results)} chunks"
                }
            
            elif function_name == "semantic_search":
                query = args["query"]
                top_k = args.get("top_k", 20)
                results = self.tools.semantic_search(query, top_k=top_k)
                
                result_id = f"semantic_{len(tool_results['semantic_search'])}"
                tool_results["semantic_search"].append({
                    "id": result_id,
                    "query": query,
                    "results": results
                })
                
                return {
                    "result_id": result_id,
                    "success": True,
                    "chunks_found": len(results),
                    "top_score": results[0].score if results else 0.0,
                    "message": f"Semantic search returned {len(results)} chunks"
                }
            
            elif function_name == "hybrid_search":
                query = args["query"]
                top_k = args.get("top_k", 20)
                bm25_weight = args.get("bm25_weight", 0.5)
                semantic_weight = args.get("semantic_weight", 0.5)
                results = self.tools.hybrid_search(
                    query,
                    top_k=top_k,
                    bm25_weight=bm25_weight,
                    semantic_weight=semantic_weight
                )
                
                result_id = f"hybrid_{len(tool_results['hybrid_search'])}"
                tool_results["hybrid_search"].append({
                    "id": result_id,
                    "query": query,
                    "results": results
                })
                
                return {
                    "result_id": result_id,
                    "success": True,
                    "chunks_found": len(results),
                    "top_score": results[0].score if results else 0.0,
                    "message": f"Hybrid search returned {len(results)} chunks"
                }
            
            elif function_name == "rrf_fusion":
                result_ids = args["result_ids"]
                rrf_k = args.get("rrf_k", 60)
                top_k = args.get("top_k", 20)
                
                # Collect all results to fuse
                all_results = []
                for rid in result_ids:
                    for category in ["bm25_search", "semantic_search", "hybrid_search"]:
                        for stored in tool_results[category]:
                            if stored["id"] == rid:
                                all_results.extend(stored["results"])
                
                # Apply RRF fusion
                fused = self._fuse_results(all_results, rrf_k=rrf_k)
                final_results = fused[:top_k]
                
                result_id = f"fusion_{len(tool_results['rrf_fusion'])}"
                tool_results["rrf_fusion"].append({
                    "id": result_id,
                    "source_ids": result_ids,
                    "results": final_results
                })
                
                return {
                    "result_id": result_id,
                    "success": True,
                    "chunks_found": len(final_results),
                    "unique_chunks": len(set(r.chunk_id for r in final_results)),
                    "message": f"Fused {len(result_ids)} result sets into {len(final_results)} chunks"
                }
            
            else:
                return {"success": False, "error": f"Unknown tool: {function_name}"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _aggregate_tool_results(self, tool_results: Dict, state: AgenticRAGState) -> tuple:
        """
        Aggregate results from all tool calls into final retrieval results
        """
        # Priority: Use fusion results if available, otherwise combine all search results
        if tool_results["rrf_fusion"]:
            # Use the last fusion result
            final_results = tool_results["rrf_fusion"][-1]["results"]
        else:
            # Combine all search results and apply fusion
            all_results = []
            for category in ["bm25_search", "semantic_search", "hybrid_search"]:
                for stored in tool_results[category]:
                    all_results.extend(stored["results"])
            
            final_results = self._fuse_results(all_results, rrf_k=60)[:20]
        
        # Build metadata
        metadata = {
            "mode": "tool_calling",
            "range_routing_calls": len(tool_results["range_routing"]),
            "bm25_calls": len(tool_results["bm25_search"]),
            "semantic_calls": len(tool_results["semantic_search"]),
            "hybrid_calls": len(tool_results["hybrid_search"]),
            "fusion_calls": len(tool_results["rrf_fusion"]),
            "total_chunks_retrieved": len(final_results),
            "unique_chunks": len(set(r.chunk_id for r in final_results)),
            "range_chunks": sum(r["count"] for r in tool_results["range_routing"])
        }
        
        return final_results, metadata
    
    def _process_with_llm_planning(self, state: AgenticRAGState) -> dict:
        """
        Use LLM to create detailed retrieval plan (optional, for complex scenarios)
        """
        question = state["question"]
        question_type = state.get("question_type", "general")
        retrieval_strategies = state.get("retrieval_strategies", ["hybrid"])
        query_candidates = state.get("query_candidates", [])
        
        # Get retrieval hints from messages (stored by QueryPlanner)
        messages = state.get("messages", [])
        retrieval_hints = []
        for msg in messages:
            if msg.startswith("Retrieval Hints:"):
                hint_part = msg.replace("Retrieval Hints:", "").strip()
                retrieval_hints = [h.strip() for h in hint_part.split(";") if h.strip()]
                break
        
        # Build prompt
        prompt = build_retrieval_router_prompt(
            question=question,
            question_type=question_type,
            retrieval_strategies=retrieval_strategies,
            query_candidates=[
                {
                    "query": qc.get("query") if isinstance(qc, dict) else qc.query,
                    "query_type": qc.get("query_type") if isinstance(qc, dict) else qc.query_type,
                    "weight": qc.get("weight", 1.0) if isinstance(qc, dict) else qc.weight
                }
                for qc in query_candidates
            ],
            retrieval_hints=retrieval_hints
        )
        
        # Call LLM for execution plan
        response = self.client.beta.chat.completions.parse(
            model=self.config.azure_deployment_name,
            messages=[
                {"role": "system", "content": RETRIEVAL_ROUTER_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            response_format=RetrievalRouterDecision,
            temperature=self.config.agent_temperature
        )
        
        decision = response.choices[0].message.parsed
        
        # Execute based on LLM's plan
        return self._execute_plan(state, decision)
    
    def _process_direct_execution(self, state: AgenticRAGState) -> dict:
        """
        Execute retrieval directly based on Orchestrator's strategies (default mode)
        """
        if self.tools is None:
            raise ValueError("RetrievalTools not initialized!")
        
        retrieval_strategies = state.get("retrieval_strategies", ["hybrid"])
        query_candidates = state.get("query_candidates", [])
        question_keywords = state.get("question_keywords", [])
        
        # Extract CPT codes from keywords for range routing
        from ..utils.keyword_parser import extract_cpt_codes, has_cpt_codes
        
        cpt_codes = extract_cpt_codes(question_keywords)
        
        # Determine if range routing is needed
        apply_range_routing = "range_routing" in retrieval_strategies and has_cpt_codes(question_keywords)
        
        # Execute retrieval
        results, metadata = self._execute_retrieval(
            query_candidates=query_candidates,
            strategies=retrieval_strategies,
            cpt_codes=cpt_codes if apply_range_routing else None,
            top_k=self.config.top_k
        )
        
        return {
            "retrieved_chunks": results,
            "retrieval_metadata": metadata
        }
    
    def _execute_retrieval(
        self,
        query_candidates: List[dict],
        strategies: List[str],
        cpt_codes: List[str] = None,
        top_k: int = 20
    ) -> tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Execute retrieval with multiple queries and strategies
        
        Args:
            query_candidates: List of query candidates from QueryPlanner
            strategies: Retrieval strategies from Orchestrator
            cpt_codes: CPT codes for range routing (if applicable)
            top_k: Number of final results to return
            
        Returns:
            (results, metadata)
        """
        all_results = []
        metadata = {
            "strategies_used": strategies,
            "num_queries": len(query_candidates),
            "range_routing_applied": cpt_codes is not None,
            "cpt_codes_filtered": cpt_codes if cpt_codes else [],
            "total_chunks_retrieved": 0
        }
        
        # Step 1: Range routing pre-filter (if applicable)
        range_chunk_ids = set()
        if cpt_codes:
            for code in cpt_codes:
                try:
                    chunk_ids = self.tools.range_routing(int(code), limit=300)
                    range_chunk_ids.update(chunk_ids)
                except:
                    pass  # Skip invalid codes
            metadata["range_chunks_count"] = len(range_chunk_ids)
        
        # Step 2: Execute retrieval for each query candidate
        for qc in query_candidates:
            query_text = qc.get("query") if isinstance(qc, dict) else qc.query
            query_weight = qc.get("weight", 1.0) if isinstance(qc, dict) else qc.weight
            
            # Determine which strategies to use for this query
            # Remove range_routing from strategies (already applied as pre-filter)
            active_strategies = [s for s in strategies if s != "range_routing"]
            
            for strategy in active_strategies:
                if strategy == "bm25":
                    results = self.tools.bm25_search(query_text, top_k=top_k * 2)
                elif strategy == "semantic":
                    results = self.tools.semantic_search(query_text, top_k=top_k * 2)
                elif strategy == "hybrid":
                    results = self.tools.hybrid_search(query_text, top_k=top_k * 2)
                else:
                    continue
                
                # Weight results by query candidate weight
                for r in results:
                    r.score *= query_weight
                    # Boost if in range routing results
                    if cpt_codes and r.chunk_id in range_chunk_ids:
                        r.score *= 1.5
                
                all_results.extend(results)
        
        # Step 3: Fuse results with RRF
        fused_results = self._fuse_results(all_results, rrf_k=self.config.rrf_k)
        
        # Take top_k
        final_results = fused_results[:top_k]
        
        metadata["total_chunks_retrieved"] = len(final_results)
        metadata["unique_chunks_before_fusion"] = len(set(r.chunk_id for r in all_results))
        
        return final_results, metadata
    
    def _fuse_results(
        self,
        results: List[RetrievalResult],
        rrf_k: int = 60
    ) -> List[RetrievalResult]:
        """
        Fuse results from multiple retrieval calls using RRF
        
        Args:
            results: All retrieval results from multiple queries/strategies
            rrf_k: RRF k parameter
            
        Returns:
            Deduplicated and re-ranked results
        """
        # Group by chunk_id and aggregate scores
        chunk_scores = {}
        chunk_data = {}
        
        for r in results:
            if r.chunk_id not in chunk_scores:
                chunk_scores[r.chunk_id] = 0.0
                chunk_data[r.chunk_id] = r
            
            # RRF-style score aggregation (simplified)
            chunk_scores[r.chunk_id] += r.score
        
        # Sort by aggregated score
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Convert back to RetrievalResult
        fused_results = []
        for chunk_id, score in sorted_chunks:
            result = chunk_data[chunk_id]
            result.score = score  # Update with fused score
            fused_results.append(result)
        
        return fused_results
    
    def _execute_plan(self, state: AgenticRAGState, decision: RetrievalRouterDecision) -> dict:
        """
        Execute retrieval based on LLM's execution plan
        
        This is the advanced mode where LLM creates a detailed plan
        """
        # TODO: Implement detailed execution based on decision schema
        # For now, fall back to direct execution
        return self._process_direct_execution(state)
         