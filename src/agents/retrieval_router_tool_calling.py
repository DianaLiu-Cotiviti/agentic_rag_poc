"""
Tool Calling Retrieval Router - LLM-driven tool calling

Characteristics:
- 5-15 LLM calls (multi-turn conversation)
- LLM dynamically adjusts strategy based on intermediate results
- Fully agentic and intelligent
- Slow (~10 seconds)
- Expensive ($0.05-0.10)
"""

import json
import logging
import sys

# Initialize logger for tool calling router
logger = logging.getLogger("agenticrag.retrieval_router_tool_calling")
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

from typing import List, Dict, Any
from ..state import AgenticRAGState, RetrievalResult, SearchGuidance
from ..prompts.search_guidance_templates import (
    get_ncci_semantic_guidance,
    get_cpt_definition_semantic_guidance,
    get_modifier_semantic_guidance,
    get_billing_policy_semantic_guidance,
    get_ncci_hybrid_guidance,
    get_cpt_definition_hybrid_guidance,
    get_modifier_hybrid_guidance
)
from ..prompts.retrieval_router_prompts import (
    build_tool_calling_prompt,
    RETRIEVAL_ROUTER_SYSTEM_MESSAGE
)


class ToolCallingRetrievalRouter:
    """
    Tool Calling Mode - LLM directly calls tools
    
    Execution Flow:
    1. Build tool definitions (5 tools)
    2. LLM sees the question and tool list
    3. Multi-turn dialogue:
       - LLM decides which tool to call
       - Agent executes tool
       - Return results to LLM
       - LLM decides next step after seeing results
       - Loop until LLM says "done"
    4. Aggregate all tool call results
    """
    
    def __init__(self, config, tools, client=None):
        """
        Args:
            config: Configuration object with Azure OpenAI settings
            tools: RetrievalTools instance
            client: Azure OpenAI client (optional, will use config.client if not provided)
        """
        self.config = config
        self.tools = tools
        self.client = client if client is not None else getattr(config, 'client', None)
        self.max_tool_iterations = 10
    
    def process(self, state: AgenticRAGState) -> dict:
        """
        LLM-driven tool calling mode
        
        Supports retry mode: uses refined_queries when retry_count > 0
        
        Args:
            state: Contains question, query_candidates/refined_queries, retry_count, keep_chunks
            
        Returns:
            dict: Contains retrieved_chunks and retrieval_metadata
        """
        question = state["question"]
        question_type = state.get("question_type", "general")
        retrieval_strategies = state.get("retrieval_strategies", ["hybrid"])
        question_keywords = state.get("question_keywords", [])
        
        # RETRY MODE: Check if this is a retry round
        retry_count = state.get("retry_count", 0)
        
        # Use refined_queries if in retry mode, otherwise use query_candidates
        if retry_count > 0 and state.get("refined_queries"):
            # Retry mode: use refined queries from Query Refiner
            query_candidates = state.get("refined_queries", [])
            logger.info(f"\n🔄 RETRY MODE (Round {retry_count}) - Tool Calling with {len(query_candidates)} refined queries")
            
            # Extract retrieval hints from refined queries (Query Refiner's hints)
            retrieval_hints = [
                rq.get("retrieval_hint") 
                for rq in query_candidates 
                if isinstance(rq, dict) and rq.get("retrieval_hint")
            ]
            if retrieval_hints:
                logger.info(f"   Using {len(retrieval_hints)} retrieval hints from Query Refiner")
        else:
            # Initial mode: use query candidates from Query Planner
            query_candidates = state.get("query_candidates", [])
            # Get retrieval hints from Query Planner (strategy-level recommendations)
            retrieval_hints = state.get("retrieval_hints", [])
        
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
            question_keywords=question_keywords,
            retrieval_hints=retrieval_hints
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
        
        # Track detailed execution steps
        execution_log = []
        
        # Initialize client if not already set
        if not self.client:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                api_key=self.config.azure_openai_api_key,
                api_version=self.config.azure_api_version,
                azure_endpoint=self.config.azure_openai_endpoint
            )
        
        # Multi-turn conversation with tool calling
        iteration = 0
        while iteration < self.max_tool_iterations:
            iteration += 1
            
            logger.info(f"\n  🔄 Tool Calling Iteration #{iteration}")
            
            response = self.client.chat.completions.create(
                model=self.config.azure_deployment_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.config.agent_temperature,
                max_tokens=1000  # Limit output for tool calling decisions
            )
            
            assistant_message = response.choices[0].message
            messages.append(assistant_message.model_dump())
            
            # Check if LLM wants to call tools
            if assistant_message.tool_calls:
                logger.info(f"     LLM requested {len(assistant_message.tool_calls)} tool call(s)")
                
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"     → Calling: {function_name}({', '.join(f'{k}={v}' for k, v in list(function_args.items())[:2])}...)")
                    
                    # Execute tool
                    result = self._execute_tool(function_name, function_args, tool_results)
                    
                    # Log execution
                    step_log = {
                        "iteration": iteration,
                        "tool_name": function_name,
                        "arguments": function_args,
                        "chunks_returned": result.get("chunks_found", 0),
                        "result_id": result.get("result_id", "N/A")
                    }
                    execution_log.append(step_log)
                    
                    logger.info(f"       ✓ Returned {result.get('chunks_found', 0)} chunks (ID: {result.get('result_id', 'N/A')})")
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
            else:
                # LLM finished tool calling
                if assistant_message.content:
                    logger.info(f"     ✅ LLM finished: {assistant_message.content[:100]}...")
                else:
                    logger.info(f"     ✅ LLM finished tool calling")
                break
        
        # Aggregate all results
        aggregated_results, metadata = self._aggregate_tool_results(tool_results, state)
        
        # Layer 2: Limit to reasonable number for Layer 3 (avoid sending too many to cross-encoder)
        # Tool Calling mode can retrieve many chunks, so we limit here
        max_for_layer3 = 20  # Send at most 20 chunks to Layer 3 cross-encoder
        if len(aggregated_results) > max_for_layer3:
            original_count = len(aggregated_results)
            aggregated_results = aggregated_results[:max_for_layer3]
            logger.info(f"\n📊 Layer 2: Truncated {original_count} → {len(aggregated_results)} chunks for Layer 3")
        
        # RETRY MODE: Merge chunks if retry_count > 0
        retry_count = state.get("retry_count", 0)
        keep_chunks = state.get("keep_chunks", [])
        missing_aspects = state.get("missing_aspects", [])
        
        if retry_count > 0 and keep_chunks:
            logger.info(f"\n🔀 Merging chunks (Tool Calling Mode - Round {retry_count}):")
            logger.info(f"   - Keep chunks (adaptive): {len(keep_chunks)}")
            logger.info(f"   - New chunks: {len(aggregated_results)}")
            logger.info(f"   - Missing aspects: {len(missing_aspects)}")
            
            # Call merge_chunks_in_retry tool
            # Return 15-20 merged chunks, then pass to Evidence Judge Layer 3 rerank
            merged_results, merge_stats = self.tools.merge_chunks_in_retry(
                keep_chunks=keep_chunks,
                new_chunks=aggregated_results,
                missing_aspects=missing_aspects,
                quality_threshold=0.75,
                top_k=20  # Return max 20 chunks, Evidence Judge will rerank to top 10
            )
            
            final_results = merged_results
            
            logger.info(f"\n✅ Merge complete (Adaptive Selection):")
            logger.info(f"   - Final chunks for Evidence Judge: {len(final_results)}")
            logger.info(f"   - Kept from old (adaptive): {merge_stats.get('kept_old_chunks', 0)}")
            logger.info(f"   - Added from new: {merge_stats.get('added_new_chunks', 0)}")
            logger.info(f"   - Removed duplicates: {merge_stats.get('duplicates_removed', 0)}")
            logger.info(f"   - Boosted for missing aspects: {merge_stats.get('boosted_chunks', 0)}")
            logger.info(f"   → Evidence Judge will rerank to top 10")
            
            # Update metadata with merge info
            metadata.update(merge_stats)
        else:
            # Initial retrieval (no merge)
            final_results = aggregated_results
        
        # Add execution log to metadata
        metadata["execution_log"] = execution_log
        metadata["total_iterations"] = iteration
        metadata["total_tool_calls"] = len(execution_log)
        
        # Save retrieved chunks to output
        from ..utils.save_workflow_outputs import save_retrieved_chunks
        saved_path = save_retrieved_chunks(
            chunks=final_results,
            question=state.get('question', ''),
            output_dir=self.config.retrieval_output_dir,
            metadata=metadata
        )
        metadata["saved_to"] = saved_path
        
        # Extract CPT descriptions from tool results
        cpt_descriptions_dict = {}
        if tool_results.get("cpt_descriptions"):
            for desc_result in tool_results["cpt_descriptions"]:
                # desc_result is a dict like {"14301": "Adjacent tissue transfer..."}
                if isinstance(desc_result, dict):
                    cpt_descriptions_dict.update(desc_result)
        
        return {
            "retrieved_chunks": final_results,
            "retrieval_metadata": metadata,
            "cpt_descriptions": cpt_descriptions_dict if cpt_descriptions_dict else None
        }
    
    def _build_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Build OpenAI function calling tool definitions
        
        Returns:
            List of tool definitions for OpenAI API
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_cpt_description",
                    "description": "Get the full official description for a CPT/HCPCS code. Use this FIRST when you need to understand what a CPT code means before searching.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cpt_code": {
                                "type": "integer",
                                "description": "CPT or HCPCS code (e.g., 14301, 27702)"
                            }
                        },
                        "required": ["cpt_code"]
                    }
                }
            },
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
                                "description": "Maximum number of chunks to retrieve (default: 50)",
                                "default": 50
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
                                "description": "Number of results to retrieve (default: 50)",
                                "default": 50
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
                    "description": "Vector-based semantic search with optional search guidance. Best for conceptual queries, definitions, and understanding context. Use guidance_types (array) to combine multiple search focuses for comprehensive retrieval.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query text"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to retrieve (default: 50)",
                                "default": 50
                            },
                            "guidance_types": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["ncci", "cpt_definition", "modifier", "billing_policy"]
                                },
                                "description": "Array of guidance types to apply (can combine multiple): 'ncci' for NCCI policies/PTP edits, 'cpt_definition' for CPT code definitions, 'modifier' for modifier documentation, 'billing_policy' for billing policies. Examples: ['ncci'], ['cpt_definition', 'modifier'], ['ncci', 'cpt_definition', 'modifier']. Default: [] (no guidance)",
                                "default": []
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
                    "description": "Combined BM25 + semantic search with weighted fusion and optional search guidance. Best for balanced exact + conceptual matching. Use guidance_types (array) to combine multiple search focuses for comprehensive retrieval.",
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
                            },
                            "guidance_types": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["ncci", "cpt_definition", "modifier", "billing_policy"]
                                },
                                "description": "Array of guidance types to apply (can combine multiple): 'ncci' for NCCI policies/PTP edits, 'cpt_definition' for CPT code definitions, 'modifier' for modifier documentation, 'billing_policy' for billing policies. Examples: ['ncci'], ['cpt_definition', 'modifier'], ['ncci', 'cpt_definition', 'modifier']. Default: [] (no guidance)",
                                "default": []
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
        Execute tool called by LLM
        
        Args:
            function_name: Tool name (e.g., "range_routing", "bm25_search")
            args: Tool arguments from LLM
            tool_results: Storage for all tool results
            
        Returns:
            Tool execution result (to be sent back to LLM)
        """
        try:
            if function_name == "get_cpt_description":
                cpt_code = args.get("cpt_code")
                description = self.tools.get_cpt_description(cpt_code)
                
                result = {
                    "cpt_code": cpt_code,
                    "description": description if description else f"No description found for CPT code {cpt_code}",
                    "has_description": bool(description)
                }
                
                # Store in tool_results (not used for final retrieval, but for context)
                tool_results["cpt_descriptions"] = tool_results.get("cpt_descriptions", [])
                tool_results["cpt_descriptions"].append(result)
                
                return result
            
            elif function_name == "range_routing":
                cpt_code = args["cpt_code"]
                limit = args.get("limit", 50)
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
                top_k = args.get("top_k", 50)
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
                top_k = args.get("top_k", 50)
                guidance_types = args.get("guidance_types", [])
                
                # Generate combined guidance from multiple types
                guidance = None
                if guidance_types:
                    # Extract CPT codes and modifiers from query for guidance generation
                    import re
                    cpt_codes = re.findall(r'\b\d{5}\b', query)
                    modifiers = re.findall(r'\bmodifier\s+(\d{2})\b|\b(59|25|LT|RT|XE|XP|XS|XU)\b', query)
                    modifiers = [m[0] or m[1] for m in modifiers if m[0] or m[1]]
                    
                    # Combine guidance texts from all requested types
                    guidance_texts = []
                    all_boost_terms = list(set(cpt_codes + modifiers))  # Deduplicate
                    
                    for guidance_type in guidance_types:
                        if guidance_type == "ncci":
                            guidance_texts.append(get_ncci_semantic_guidance(cpt_codes))
                        elif guidance_type == "cpt_definition":
                            guidance_texts.append(get_cpt_definition_semantic_guidance(cpt_codes))
                        elif guidance_type == "modifier":
                            guidance_texts.append(get_modifier_semantic_guidance(modifiers, cpt_codes if cpt_codes else None))
                        elif guidance_type == "billing_policy":
                            guidance_texts.append(get_billing_policy_semantic_guidance("billing policy"))
                    
                    if guidance_texts:
                        # Combine all guidance texts with clear separators
                        combined_guidance = "\n\n---\n\n".join(guidance_texts)
                        guidance = SearchGuidance(
                            semantic_guidance=combined_guidance,
                            boost_terms=all_boost_terms
                        )
                
                results = self.tools.semantic_search(query, top_k=top_k, guidance=guidance)
                
                result_id = f"semantic_{len(tool_results['semantic_search'])}"
                tool_results["semantic_search"].append({
                    "id": result_id,
                    "query": query,
                    "guidance_types": guidance_types,
                    "results": results
                })
                
                return {
                    "result_id": result_id,
                    "success": True,
                    "chunks_found": len(results),
                    "top_score": results[0].score if results else 0.0,
                    "guidance_applied": ", ".join(guidance_types) if guidance_types else "none",
                    "message": f"Semantic search returned {len(results)} chunks (guidance: {', '.join(guidance_types) if guidance_types else 'none'})"
                }
            
            elif function_name == "hybrid_search":
                query = args["query"]
                top_k = args.get("top_k", 20)
                bm25_weight = args.get("bm25_weight", 0.5)
                semantic_weight = args.get("semantic_weight", 0.5)
                guidance_types = args.get("guidance_types", [])
                
                # Generate combined guidance from multiple types
                guidance = None
                if guidance_types:
                    # Extract CPT codes and modifiers from query
                    import re
                    cpt_codes = re.findall(r'\b\d{5}\b', query)
                    modifiers = re.findall(r'\bmodifier\s+(\d{2})\b|\b(59|25|LT|RT|XE|XP|XS|XU)\b', query)
                    modifiers = [m[0] or m[1] for m in modifiers if m[0] or m[1]]
                    
                    # Combine guidance from all types (hybrid returns dict)
                    guidance_texts = []
                    all_boost_terms = list(set(cpt_codes + modifiers))
                    combined_metadata_filters = {}
                    
                    for guidance_type in guidance_types:
                        guidance_dict = {}
                        if guidance_type == "ncci":
                            guidance_dict = get_ncci_hybrid_guidance(cpt_codes)
                        elif guidance_type == "cpt_definition":
                            guidance_dict = get_cpt_definition_hybrid_guidance(cpt_codes)
                        elif guidance_type == "modifier":
                            guidance_dict = get_modifier_hybrid_guidance(modifiers, cpt_codes if cpt_codes else None)
                        
                        if guidance_dict:
                            guidance_texts.append(guidance_dict.get("semantic_guidance", ""))
                            all_boost_terms.extend(guidance_dict.get("boost_terms", []))
                            # Merge metadata filters (combining doc_types)
                            for key, value in guidance_dict.get("metadata_filters", {}).items():
                                if key in combined_metadata_filters:
                                    if isinstance(value, list):
                                        combined_metadata_filters[key].extend(value)
                                else:
                                    combined_metadata_filters[key] = value if isinstance(value, list) else [value]
                    
                    if guidance_texts:
                        # Deduplicate boost terms
                        all_boost_terms = list(set(all_boost_terms))
                        # Deduplicate metadata filter lists
                        for key in combined_metadata_filters:
                            if isinstance(combined_metadata_filters[key], list):
                                combined_metadata_filters[key] = list(set(combined_metadata_filters[key]))
                        
                        combined_guidance = "\n\n---\n\n".join(guidance_texts)
                        guidance = SearchGuidance(
                            semantic_guidance=combined_guidance,
                            boost_terms=all_boost_terms,
                            metadata_filters=combined_metadata_filters
                        )
                
                results = self.tools.hybrid_search(
                    query,
                    top_k=top_k,
                    bm25_weight=bm25_weight,
                    semantic_weight=semantic_weight,
                    guidance=guidance
                )
                
                result_id = f"hybrid_{len(tool_results['hybrid_search'])}"
                tool_results["hybrid_search"].append({
                    "id": result_id,
                    "query": query,
                    "guidance_types": guidance_types,
                    "results": results
                })
                
                return {
                    "result_id": result_id,
                    "success": True,
                    "chunks_found": len(results),
                    "top_score": results[0].score if results else 0.0,
                    "guidance_applied": ", ".join(guidance_types) if guidance_types else "none",
                    "message": f"Hybrid search returned {len(results)} chunks (guidance: {', '.join(guidance_types) if guidance_types else 'none'})"
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
        Aggregate all tool call results
        
        Args:
            tool_results: All tool call results
            state: Current state
            
        Returns:
            (final_results, metadata)
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
        
        # Build strategies_used list for reporting
        strategies_used = []
        if tool_results["range_routing"]:
            strategies_used.append("range_routing")
        if tool_results["bm25_search"]:
            strategies_used.append("bm25")
        if tool_results["semantic_search"]:
            strategies_used.append("semantic")
        if tool_results["hybrid_search"]:
            strategies_used.append("hybrid")
        if tool_results["rrf_fusion"]:
            strategies_used.append("rrf_fusion")
        
        # Build metadata
        metadata = {
            "mode": "tool_calling",
            "strategies_used": strategies_used,  # Record strategies used by LLM
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
    
    def _fuse_results(
        self,
        results: List[RetrievalResult],
        rrf_k: int = 60
    ) -> List[RetrievalResult]:
        """
        Fuse multiple retrieval results using RRF
        
        Args:
            results: All retrieval results
            rrf_k: RRF k parameter
            
        Returns:
            Fused and re-ranked results
        """
        # Group by chunk_id and aggregate scores
        chunk_scores = {}
        chunk_data = {}
        
        for r in results:
            if r.chunk_id not in chunk_scores:
                chunk_scores[r.chunk_id] = 0.0
                chunk_data[r.chunk_id] = r
            
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
            result.score = score
            fused_results.append(result)
        
        return fused_results
