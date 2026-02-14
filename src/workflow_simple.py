"""
Simple Agentic RAG Workflow - Non-Iteration Version
====================================================

This is a simplified workflow for validating the basic agent pipeline:
User Query → Orchestrator → Query Planner → Retrieval Router → Evidence Judge 
         → (if sufficient) Answer Generator → END
         → (if insufficient) END (reserved for future Query Refiner)

Features:
- Evidence Judge sufficiency determination
- Conditional edge: sufficient → answer, insufficient → END
- Answer Generator: generates answer based on top 10 chunks

Used for testing that each agent is correctly connected and functioning.
"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END

from .state import AgenticRAGState
from .config import AgenticRAGConfig
from .memory import WorkflowMemory
from .agents_coordinator import AgenticRAGAgents
from .tools.retrieval_tools import RetrievalTools
from .tools.build_indexes import ensure_all_indexes
from .utils.save_workflow_outputs import save_query_candidates, save_retrieved_chunks, save_final_answer
import logging
import sys
logger = logging.getLogger("agenticrag.workflow_simple")
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class SimpleAgenticRAGWorkflow:
    """
    Simplified Agentic RAG Workflow
    
    Flow:
    1. Orchestrator → Analyze question, select retrieval mode
    2. Query Planner → Generate query candidates (if in planning mode)
    3. Retrieval Router → Execute retrieval, return top 15 chunks
    4. Evidence Judge → Assess quality (new chunk formatting + LLM summarization)
    5. END → Return results
    
    Usage:
        config = AgenticRAGConfig.from_env()
        workflow = SimpleAgenticRAGWorkflow(config)
        result = workflow.run("What is CPT code 14301?")
    """
    
    def __init__(self, config: AgenticRAGConfig = None, enable_memory: bool = True):
        """
        Initialize workflow
        
        Args:
            config: Configuration object, loads from environment variables if None
            enable_memory: Whether to enable memory saving functionality (default True)
        """
        self.config = config or AgenticRAGConfig.from_env()
        
        # Ensure all indexes are built before initializing agents
        logger.info("\n🔧 Preprocessing: Ensuring all indexes are built...")
        ensure_all_indexes(
            chunks_path=self.config.chunks_path,
            range_index_path=self.config.range_index_path,
            bm25_index_path=self.config.bm25_index_path,
            chroma_dir=self.config.chroma_db_path,
            config=self.config  # Pass config for embedding client
        )
        
        self.agents = AgenticRAGAgents(self.config)
        self.tools = RetrievalTools(self.config)
        self.graph = self._build_graph()
        
        # Memory manager
        self.enable_memory = enable_memory
        if self.enable_memory:
            self.memory = WorkflowMemory(memory_dir=self.config.memory_dir)
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        
        # Create graph
        workflow = StateGraph(AgenticRAGState)
        
        # Add nodes
        workflow.add_node("orchestrator", self._orchestrator_node)
        workflow.add_node("query_planner", self._query_planner_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("evidence_judge", self._evidence_judge_node)
        workflow.add_node("answer_generator", self._answer_generator_node)  # NEW
        workflow.add_node("query_refiner", self._query_refiner_node)  # NEW: Retry logic
        
        # Set entry point
        workflow.set_entry_point("orchestrator")
        
        # Add edges
        workflow.add_edge("orchestrator", "query_planner")
        workflow.add_edge("query_planner", "retrieval")
        workflow.add_edge("retrieval", "evidence_judge")
        
        # Conditional edge: Evidence Judge → Answer Generator / Query Refiner / END
        workflow.add_conditional_edges(
            "evidence_judge",
            self._should_retry_or_answer,
            {
                "answer": "answer_generator",  # Evidence is sufficient → generate answer
                "retry": "query_refiner",  # Evidence is insufficient + retry < max → refine queries
                "end": END  # Evidence is insufficient + retry >= max → end
            }
        )
        
        # Query Refiner → Retrieval (retry loop, skip Query Planner)
        workflow.add_edge("query_refiner", "retrieval")
        
        # Answer Generator → END
        workflow.add_edge("answer_generator", END)
        
        return workflow.compile()
    
    # ========== Node Functions ==========
    
    def _orchestrator_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """
        Orchestrator node
        
        Responsibilities:
        1. Analyze question type (cpt_code_lookup, billing_compatibility, etc.)
        2. Select retrieval mode (direct, planning, tool_calling)
        3. Set max_retry (not used here, but still set)
        
        Note: Only executed in initial round, skipped during retry
        """
        logger.info("\n" + "="*80)
        logger.info("🎯🎯 STEP 1: ORCHESTRATOR [ROUND 1 - INITIAL]")
        logger.info("="*80)
        
        result = self.agents.orchestrator_node(state)
        
        logger.info(f"Question Type: {result.get('question_type')}")
        logger.info(f"Complexity: {result.get('question_complexity')}")
        logger.info(f"Strategy Hints: {result.get('retrieval_strategies')}")
        logger.info(f"Reasoning: {result.get('orchestrator_reasoning', 'N/A')[:200]}...")
        
        state.update(result)
        return state
    
    def _query_planner_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """
        Query Planner node
        
        Responsibilities:
        1. Generate query candidates (if in planning or tool_calling mode)
        2. Direct mode skips this step (or generates minimal queries)
        3. Save query candidates to output/queries
        
        Note: Only executed in initial round, skipped during retry
        """
        logger.info("\n" + "="*80)
        logger.info("📋📋 STEP 2: QUERY PLANNER [ROUND 1 - INITIAL]")
        logger.info("="*80)
        
        result = self.agents.query_planner_node(state)
        
        query_candidates = result.get('query_candidates', [])
        for i, qc in enumerate(query_candidates, 1):
            # Only output sub query string
            if isinstance(qc, dict):
                query_text = qc.get('query', str(qc))
            else:
                query_text = getattr(qc, 'query', str(qc))
            logger.info(f"  {i}. {query_text}")
        
        # Save query candidates to output/queries (silent save, no path output)
        if query_candidates:
            from .utils.save_workflow_outputs import save_query_candidates
            save_query_candidates(
                query_candidates=query_candidates,
                question=state.get('question', ''),
                output_dir=self.config.query_output_dir,
                metadata={
                    'question_type': state.get('question_type'),
                    'question_complexity': state.get('question_complexity'),
                    'retrieval_strategies': state.get('retrieval_strategies'),
                    'mode': self.config.retrieval_mode
                }
            )
        
        state.update(result)
        return state
    
    def _retrieval_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """
        Retrieval Router node
        
        Responsibilities:
        1. Execute corresponding retrieval strategy based on mode
        2. Return top 15-20 chunks (already fused)
        """
        retry_count = state.get("retry_count", 0)
        max_retry = state.get("max_retry", 2)
        total_round = retry_count + 1  # Round 1, 2, 3, ...
        round_label = f"[ROUND {total_round} - INITIAL]" if retry_count == 0 else f"[ROUND {total_round} - RETRY #{retry_count}/{max_retry}]"
        
        logger.info("\n" + "="*80)
        logger.info(f"🔍🔍 STEP 3: RETRIEVAL ROUTER {round_label}")
        logger.info("="*80)
        
        # Mode comes from config, not from state
        mode = self.config.retrieval_mode
        logger.info(f"Mode: {mode}")
        
        result = self.agents.retrieval_router_node(state, self.tools)
        
        chunks = result.get('retrieved_chunks', [])
        metadata = result.get('retrieval_metadata', {})
        
        # Show detailed execution based on mode
        execution_log = metadata.get('execution_log', [])
        if execution_log:
            # Tool calling mode - show tool call details
            logger.info(f"\n📊 Tool Calling Execution Summary:")
            logger.info(f"   Total tool calls: {metadata.get('total_tool_calls', 0)}")
            logger.info(f"\n  Detailed execution log:")
            for i, log in enumerate(execution_log, 1):
                logger.info(f"    Call #{i}: {log['tool_name']}(")
                args_str = ", ".join(f"{k}={v}" for k, v in list(log['arguments'].items())[:2])
                logger.info(f"{args_str}...) → {log['chunks_returned']} chunks")
        else:
            # Planning or direct mode
            per_query_stats = metadata.get('per_query_stats', [])
            if per_query_stats:
                logger.info(f"\nPer-query execution details:")
                for stats in per_query_stats:
                    logger.info(f"\n  Query #{stats['query_index']}: {stats['strategy']}")
                    logger.info(f"    Text: {stats['query_text']}")
                    logger.info(f"    Weight: {stats['weight']:.2f}")
                    logger.info(f"    Tools called: {', '.join(stats['tools_called'])}")
                    logger.info(f"    Chunks retrieved: {stats['chunks_retrieved']}")
            else:
                # Direct mode - just show strategies
                strategies_used = metadata.get('strategies_used', [])
                if strategies_used:
                    logger.info(f"\nStrategies executed:")
                    for strategy in strategies_used:
                        logger.info(f"  • {strategy}")
        
        logger.info(f"\nFinal results:")
        logger.info(f"  Retrieved chunks: {len(chunks)}")
        if chunks:
            logger.info(f"  Top chunk score: {chunks[0].score:.4f}")
            logger.info(f"  Lowest chunk score: {chunks[-1].score:.4f}")
        
        state.update(result)
        return state
    
    def _evidence_judge_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """
        Evidence Judge node
        
        Responsibilities:
        1. Use new three-layer chunk formatting strategy
        2. LLM batch summarization for truncated parts
        3. Assess coverage, specificity
        4. Return is_sufficient judgment
        """
        retry_count = state.get("retry_count", 0)
        max_retry = state.get("max_retry", 2)
        total_round = retry_count + 1  # Round 1, 2, 3, ...
        round_label = f"[ROUND {total_round} - INITIAL]" if retry_count == 0 else f"[ROUND {total_round} - RETRY #{retry_count}/{max_retry}]"
        
        logger.info("\n" + "="*80)
        logger.info(f"⚖️⚖️  STEP 4: EVIDENCE JUDGE {round_label}")
        logger.info("="*80)
        
        result = self.agents.evidence_judge_node(state)
        
        assessment = result.get('evidence_assessment', {})
        logger.info(f"Is Sufficient: {assessment.get('is_sufficient')}")
        logger.info(f"Coverage Score: {assessment.get('coverage_score', 0):.2f}")
        logger.info(f"Specificity Score: {assessment.get('specificity_score', 0):.2f}")
        logger.info(f"Has Contradiction: {assessment.get('has_contradiction')}")
        if assessment.get('missing_aspects'):
            logger.info(f"Missing Aspects: {assessment.get('missing_aspects')}")
        logger.info(f"\nReasoning:\n{assessment.get('reasoning', 'N/A')[:300]}...")
        
        state.update(result)
        return state
    
    def _should_retry_or_answer(
        self, 
        state: AgenticRAGState
    ) -> Literal["answer", "retry", "end"]:
        """
        Conditional edge: Determine whether to generate answer, retry, or end
        
        Decision Logic by Mode:
        
        **Direct Mode** (0 LLM calls):
        - Generate answer directly, no sufficiency check, no retry
        - Rationale: Direct mode uses fixed hybrid+RRF strategy, results are stable and reliable
        
        **Planning Mode** (1 LLM call):
        - Check evidence sufficiency
        - If sufficient → "answer" (go to Answer Generator)
        - If insufficient + retry < max → "retry" (go to Query Refiner)
        - If insufficient + retry >= max → "end"
        
        **Tool Calling Mode** (5-15 LLM calls):
        - Check evidence sufficiency
        - If sufficient → "answer" (go to Answer Generator)
        - If insufficient + retry < max → "retry" (go to Query Refiner)
        - If insufficient + retry >= max → "end"
        
        Args:
            state: Current state with evidence_assessment, retrieval_metadata, retry_count
            
        Returns:
            "answer" | "retry" | "end"
        """
        # Get retrieval mode from metadata
        retrieval_metadata = state.get("retrieval_metadata", {})
        mode = retrieval_metadata.get("mode", self.config.retrieval_mode)
        
        # Direct mode: generate answer directly, no sufficiency check, no retry
        if mode == "direct":
            logger.info("\n✅ Direct Mode: Skipping sufficiency check → Proceeding to Answer Generator")
            logger.info("   (Direct mode uses stable hybrid+RRF strategy, always generates answer)")
            return "answer"
        
        # Planning & Tool Calling modes: check sufficiency + retry logic
        assessment = state.get("evidence_assessment")
        
        # Safety check
        if not assessment:
            logger.info(f"\n⚠️  No evidence assessment found for {mode} mode, ending workflow")
            return "end"
        
        is_sufficient = assessment.get("is_sufficient", False)
        retry_count = state.get("retry_count", 0)
        max_retry = state.get("max_retry", 2)  # Default max_retry = 2
        
        if is_sufficient:
            logger.info(f"\n✅ {mode.title()} Mode: Evidence is SUFFICIENT → Proceeding to Answer Generator")
            if retry_count > 0:
                logger.info(f"   (Achieved sufficiency after {retry_count} retry rounds)")
            return "answer"
        
        # Evidence is insufficient
        if retry_count >= max_retry:
            logger.info(f"\n❌ {mode.title()} Mode: Evidence is INSUFFICIENT + Max retries reached ({retry_count}/{max_retry})")
            logger.info("   → Ending workflow")
            return "end"
        
        # Can retry
        logger.info(f"\n🔄 {mode.title()} Mode: Evidence is INSUFFICIENT → Retry {retry_count + 1}/{max_retry}")
        logger.info(f"   Missing aspects: {len(state.get('missing_aspects', []))}")
        return "retry"
    
    def _query_refiner_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """
        Query Refiner node
        
        Responsibilities:
        1. Analyze missing_aspects from Evidence Judge
        2. Generate refined queries targeting specific gaps
        3. Select top 3 chunks to preserve (keep_chunks)
        4. Increment retry_count
        """
        retry_count = state.get("retry_count", 0)
        max_retry = state.get("max_retry", 2)  # Default max_retry = 2
        current_retry = retry_count + 1  # This is the retry being executed now
        total_round = current_retry + 1  # Round 2, 3, 4, ... (after initial Round 1)
        
        logger.info("\n" + "#"*80)
        logger.info(f"🔄🔄 ROUND {total_round} - RETRY #{current_retry}/{max_retry} - QUERY REFINER")
        logger.info("#"*80)
        
        result = self.agents.query_refiner_node(state)
        
        refined_queries = result.get('refined_queries', [])
        logger.info(f"\n✅ Generated {len(refined_queries)} refined queries")
        for i, rq in enumerate(refined_queries, 1):
            logger.info(f"   {i}. {rq.get('query', 'N/A')[:100]}")
        
        keep_chunks = result.get('keep_chunks', [])
        logger.info(f"\n📦 Preserving {len(keep_chunks)} top chunks for merge")
        
        state.update(result)
        return state
    
    def _answer_generator_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """
        Answer Generator node
        
        Responsibilities:
        1. Receive top 10 chunks (validated as sufficient)
        2. Generate answer based on original question
        3. Return final_answer (with citations, key_points, confidence)
        
        Note: May execute in any round (once evidence is sufficient)
        """
        retry_count = state.get("retry_count", 0)
        max_retry = state.get("max_retry", 2)
        total_round = retry_count + 1  # Round 1, 2, 3, ...
        round_label = f"[ROUND {total_round} - INITIAL]" if retry_count == 0 else f"[ROUND {total_round} - RETRY #{retry_count}/{max_retry} SUCCESS]"
        
        logger.info("\n" + "="*80)
        logger.info(f"💬💬 STEP 5: ANSWER GENERATOR {round_label}")
        logger.info("="*80)
        logger.info("Calling answer_generator_node...")
        result = self.agents.answer_generator_node(state)
        final_answer = result.get('final_answer', {})
        if final_answer.get('limitations'):
            logger.info(f"Limitations: {final_answer.get('limitations')}")
        # Save Answer Generator output to output/responses/
        retrieval_metadata = state.get('retrieval_metadata', {})
        metadata = {
            'mode': retrieval_metadata.get('mode', self.config.retrieval_mode),
            'num_chunks': len(state.get('retrieved_chunks', [])),
            'is_sufficient': state.get('evidence_assessment', {}).get('is_sufficient', True)
        }
        save_path = save_final_answer(
            final_answer=final_answer,
            question=state.get('question', ''),
            output_dir=self.config.response_output_dir,
            metadata=metadata
        )
        state.update(result)
        return state
    
    # ========== Public Interface ==========
    
    def run(self, question: str, cpt_code: int = None, context: str = None) -> Dict[str, Any]:
        """
        Run simplified workflow, supports retry_rounds aggregation for UI display, and records log_text for each round
        """
        import io
        import contextlib
        log_buffer = io.StringIO()
        def get_log():
            return log_buffer.getvalue()

        logger.info("\n" + "🚀" + "="*78 + "🚀")
        logger.info("Starting Agentic RAG Workflow")
        logger.info("🚀" + "="*78 + "🚀")
        logger.info(f"\nQuestion: {question}")
        if cpt_code:
            logger.info(f"CPT Code: {cpt_code}")

        # Initialize state
        state = AgenticRAGState(
            question=question,
            cpt_code=cpt_code,
            context=context,
        )

        retry_rounds = []
        max_retry = 2
        retry_count = 0
        mode = self.config.retrieval_mode
        finished = False
        last_log_pos = 0
        # Temporarily redirect logger's handler to log_buffer to ensure UI captures all logger.info
        old_handlers = logger.handlers[:]
        stream_handler = logging.StreamHandler(log_buffer)
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.handlers = [stream_handler]
        try:
            while not finished:
                # 1. Orchestrator (only executed in initial round)
                if retry_count == 0:
                    state = self._orchestrator_node(state)
                # 2. Query Planner (only executed in initial round)
                if retry_count == 0:
                    state = self._query_planner_node(state)
                    # Set log start position after initial round to avoid retry_rounds including STEP 1/2
                    if mode in ["planning", "tool_calling"]:
                        cur_log = get_log()
                        last_log_pos = len(cur_log)
                # 3. Retrieval (executed every time)
                state = self._retrieval_node(state)

                if mode == "direct":
                    # Direct mode: retrieval → evidence_judge → answer_generator
                    state = self._evidence_judge_node(state)
                    state = self._answer_generator_node(state)
                    finished = True
                    continue

                # 4. Evidence Judge
                state = self._evidence_judge_node(state)
                assessment = state.get("evidence_assessment", {})
                is_sufficient = assessment.get("is_sufficient", False)

                # Only record retry_rounds in planning/tool_calling when evidence is insufficient
                if mode in ["planning", "tool_calling"] and (retry_count > 0 or not is_sufficient):
                    cur_log = get_log()
                    retry_log = cur_log[last_log_pos:]
                    retry_rounds.append({
                        "log_text": retry_log,
                        "query_refiner_output": state.get("refined_queries"),
                        "retrieval_router_output": state.get("retrieved_chunks"),
                        "evidence_judge_output": assessment,
                        "answer_generator_output": None  # Only present when sufficient
                    })
                    last_log_pos = len(cur_log)

                if is_sufficient:
                    # 5. Answer Generator
                    state = self._answer_generator_node(state)
                    # Record answer_generator_output in the last retry round (only for planning/tool_calling)
                    if mode in ["planning", "tool_calling"] and retry_rounds:
                        retry_rounds[-1]["answer_generator_output"] = state.get("final_answer")
                        # Record this round's log
                        cur_log = get_log()
                        retry_log = cur_log[last_log_pos:]
                        retry_rounds[-1]["log_text"] = retry_log
                    finished = True
                else:
                    retry_count = state.get("retry_count", retry_count)
                    if retry_count >= max_retry:
                        finished = True
                    else:
                        # Enter retry, call Query Refiner
                        state = self._query_refiner_node(state)
        finally:
            logger.handlers = old_handlers
        # Aggregate retry_rounds to final result
        result = dict(state)
        if retry_rounds:
            result["retry_rounds"] = retry_rounds
        # Global log_text
        result["log_text"] = log_buffer.getvalue()
        # Add mode to result
        result["retrieval_mode"] = mode
        # Ensure final_answer field exists in direct mode
        if mode == "direct" and "final_answer" not in result:
            result["final_answer"] = state.get("final_answer")

        logger.info("\n" + "✅" + "="*78 + "✅")
        logger.info("Workflow completed successfully!")
        logger.info("✅" + "="*78 + "✅")

        # Save to memory
        if self.enable_memory:
            try:
                saved_path = self.memory.save_execution(
                    question=question,
                    final_state=result,
                    workflow_type="simple",
                    mode=self.config.retrieval_mode,
                    success=True
                )
                logger.info(f"\n💾 Workflow result saved to: {saved_path}")
            except Exception as e:
                logger.info(f"\n⚠️  Failed to save memory: {e}")

        return result
    
    def visualize(self, output_path: str = "workflow_simple.png"):
        """
        Visualize workflow graph
        
        Args:
            output_path: Output image path
        """
        try:
            from IPython.display import Image, display
            display(Image(self.graph.get_graph().draw_mermaid_png()))
        except:
            logger.info("Visualization requires IPython. Saving to file instead...")
            # Save to file
            graph_image = self.graph.get_graph().draw_mermaid_png()
            with open(output_path, "wb") as f:
                f.write(graph_image)
            logger.info(f"Graph saved to {output_path}")
