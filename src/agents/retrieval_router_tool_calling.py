"""
Tool Calling Retrieval Router - LLM驱动的工具调用

特点：
- 5-15次LLM调用（multi-turn conversation）
- LLM根据中间结果动态调整策略
- 完全agentic，智能
- 慢（~10秒）
- 贵（$0.05-0.10）
"""

import json
from typing import List, Dict, Any
from ..state import AgenticRAGState, RetrievalResult
from ..prompts.retrieval_router_prompts import (
    build_tool_calling_prompt,
    RETRIEVAL_ROUTER_SYSTEM_MESSAGE
)


class ToolCallingRetrievalRouter:
    """
    Tool Calling模式 - LLM亲自调用工具
    
    执行流程：
    1. 构建tool definitions（5个工具）
    2. LLM看到问题和工具列表
    3. Multi-turn对话：
       - LLM决定call哪个tool
       - Agent执行tool
       - 返回结果给LLM
       - LLM看到结果后决定下一步
       - 循环直到LLM说"完成"
    4. 汇总所有工具调用的结果
    """
    
    def __init__(self, config, tools):
        """
        Args:
            config: Configuration object with Azure OpenAI settings
            tools: RetrievalTools instance
        """
        self.config = config
        self.tools = tools
        self.client = config.client  # Azure OpenAI client
        self.max_tool_iterations = 10
    
    def process(self, state: AgenticRAGState) -> dict:
        """
        LLM驱动的工具调用模式
        
        Args:
            state: Contains question, question_type, retrieval_strategies, query_candidates
            
        Returns:
            dict: Contains retrieved_chunks and retrieval_metadata
        """
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
        构建OpenAI function calling的工具定义
        
        Returns:
            List of tool definitions for OpenAI API
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
        执行LLM调用的工具
        
        Args:
            function_name: Tool name (e.g., "range_routing", "bm25_search")
            args: Tool arguments from LLM
            tool_results: Storage for all tool results
            
        Returns:
            Tool execution result (to be sent back to LLM)
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
        汇总所有工具调用的结果
        
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
    
    def _fuse_results(
        self,
        results: List[RetrievalResult],
        rrf_k: int = 60
    ) -> List[RetrievalResult]:
        """
        使用RRF融合多个检索结果
        
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
