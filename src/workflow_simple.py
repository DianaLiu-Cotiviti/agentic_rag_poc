"""
Simple Agentic RAG Workflow - 无Iteration版本
==============================================

这是一个简化的workflow，用于验证基本的agent pipeline：
User Query → Orchestrator → Query Planner → Retrieval Router → Evidence Judge → END

不包含:
- Query Refiner (retry逻辑)
- Structured Extraction (最终答案生成)

用于测试每个agent是否正确连接和工作。
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END

from .state import AgenticRAGState
from .config import AgenticRAGConfig
from .memory import WorkflowMemory
from .agents_coordinator import AgenticRAGAgents
from .tools.retrieval_tools import RetrievalTools
from .tools.build_indexes import ensure_all_indexes


class SimpleAgenticRAGWorkflow:
    """
    简化的Agentic RAG Workflow
    
    流程:
    1. Orchestrator → 分析问题，选择retrieval mode
    2. Query Planner → 生成query candidates (如果是planning mode)
    3. Retrieval Router → 执行检索，返回top 15 chunks
    4. Evidence Judge → 评估质量（新的chunk formatting + LLM总结）
    5. END → 返回结果
    
    使用方法:
        config = AgenticRAGConfig.from_env()
        workflow = SimpleAgenticRAGWorkflow(config)
        result = workflow.run("What is CPT code 14301?")
    """
    
    def __init__(self, config: AgenticRAGConfig = None, enable_memory: bool = True):
        """
        初始化workflow
        
        Args:
            config: 配置对象，如果为None则从环境变量加载
            enable_memory: 是否启用memory保存功能（默认True）
        """
        self.config = config or AgenticRAGConfig.from_env()
        
        # 在初始化agents之前，确保所有indexes已构建
        print("\n🔧 Preprocessing: Ensuring all indexes are built...")
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
        
        # Memory管理器
        self.enable_memory = enable_memory
        if self.enable_memory:
            self.memory = WorkflowMemory(memory_dir=self.config.memory_dir)
    
    def _build_graph(self) -> StateGraph:
        """构建LangGraph workflow"""
        
        # 创建graph
        workflow = StateGraph(AgenticRAGState)
        
        # 添加节点
        workflow.add_node("orchestrator", self._orchestrator_node)
        workflow.add_node("query_planner", self._query_planner_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("evidence_judge", self._evidence_judge_node)
        
        # 设置入口
        workflow.set_entry_point("orchestrator")
        
        # 添加边（简单的线性流程）
        workflow.add_edge("orchestrator", "query_planner")
        workflow.add_edge("query_planner", "retrieval")
        workflow.add_edge("retrieval", "evidence_judge")
        workflow.add_edge("evidence_judge", END)
        
        return workflow.compile()
    
    # ========== Node Functions ==========
    
    def _orchestrator_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """
        Orchestrator节点
        
        职责：
        1. 分析问题类型 (cpt_code_lookup, billing_compatibility, etc.)
        2. 选择retrieval mode (direct, planning, tool_calling)
        3. 设置max_retry（这里不使用，但仍设置）
        """
        print("\n" + "="*80)
        print("🎯 Step 1: Orchestrator - Analyzing question...")
        print("="*80)
        
        result = self.agents.orchestrator_node(state)
        
        print(f"Question Type: {result.get('question_type')}")
        print(f"Complexity: {result.get('question_complexity')}")
        print(f"Strategy Hints: {result.get('retrieval_strategies')}")
        print(f"Reasoning: {result.get('orchestrator_reasoning', 'N/A')[:200]}...")
        
        state.update(result)
        return state
    
    def _query_planner_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """
        Query Planner节点
        
        职责：
        1. 生成query candidates (如果是planning或tool_calling mode)
        2. Direct mode会跳过这一步（或生成minimal queries）
        3. 保存query candidates到output/queries
        """
        print("\n" + "="*80)
        print("📋 Step 2: Query Planner - Generating query candidates...")
        print("="*80)
        
        result = self.agents.query_planner_node(state)
        
        query_candidates = result.get('query_candidates', [])
        print(f"Generated {len(query_candidates)} query candidates:")
        for i, qc in enumerate(query_candidates, 1):
            # qc is a QueryCandidate object
            query_text = qc.query if hasattr(qc, 'query') else str(qc)
            print(f"  {i}. {query_text}")
        
        # Save query candidates to output/queries
        if query_candidates:
            from .utils.save_workflow_outputs import save_query_candidates
            saved_path = save_query_candidates(
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
            print(f"💾 Query candidates saved to: {saved_path}")
        
        state.update(result)
        return state
    
    def _retrieval_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """
        Retrieval Router节点
        
        职责：
        1. 根据mode执行对应的retrieval策略
        2. 返回top 15-20 chunks（已融合）
        """
        print("\n" + "="*80)
        print("🔍 Step 3: Retrieval Router - Executing retrieval...")
        print("="*80)
        
        # Mode comes from config, not from state
        mode = self.config.retrieval_mode
        print(f"Mode: {mode}")
        
        result = self.agents.retrieval_router_node(state, self.tools)
        
        chunks = result.get('retrieved_chunks', [])
        metadata = result.get('retrieval_metadata', {})
        
        # Show detailed execution based on mode
        execution_log = metadata.get('execution_log', [])
        if execution_log:
            # Tool calling mode - show iteration details
            print(f"\n📊 Tool Calling Execution Summary:")
            print(f"   Total iterations: {metadata.get('total_iterations', 0)}")
            print(f"   Total tool calls: {metadata.get('total_tool_calls', 0)}")
            
            print(f"\n  Detailed execution log:")
            for log in execution_log:
                print(f"    Iter {log['iteration']}: {log['tool_name']}(", end="")
                args_str = ", ".join(f"{k}={v}" for k, v in list(log['arguments'].items())[:2])
                print(f"{args_str}...) → {log['chunks_returned']} chunks")
        else:
            # Planning or direct mode
            per_query_stats = metadata.get('per_query_stats', [])
            if per_query_stats:
                print(f"\nPer-query execution details:")
                for stats in per_query_stats:
                    print(f"\n  Query #{stats['query_index']}: {stats['strategy']}")
                    print(f"    Text: {stats['query_text']}")
                    print(f"    Weight: {stats['weight']:.2f}")
                    print(f"    Tools called: {', '.join(stats['tools_called'])}")
                    print(f"    Chunks retrieved: {stats['chunks_retrieved']}")
            else:
                # Direct mode - just show strategies
                strategies_used = metadata.get('strategies_used', [])
                if strategies_used:
                    print(f"\nStrategies executed:")
                    for strategy in strategies_used:
                        print(f"  • {strategy}")
        
        print(f"\nFinal results:")
        print(f"  Retrieved chunks: {len(chunks)}")
        if chunks:
            print(f"  Top chunk score: {chunks[0].score:.4f}")
            print(f"  Lowest chunk score: {chunks[-1].score:.4f}")
        
        state.update(result)
        return state
    
    def _evidence_judge_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """
        Evidence Judge节点
        
        职责：
        1. 使用新的三层chunk formatting策略
        2. LLM批量总结截断部分
        3. 评估coverage, specificity
        4. 返回is_sufficient判断
        """
        print("\n" + "="*80)
        print("⚖️  Step 4: Evidence Judge - Assessing evidence quality...")
        print("="*80)
        
        result = self.agents.evidence_judge_node(state)
        
        assessment = result.get('evidence_assessment', {})
        print(f"Is Sufficient: {assessment.get('is_sufficient')}")
        print(f"Coverage Score: {assessment.get('coverage_score', 0):.2f}")
        print(f"Specificity Score: {assessment.get('specificity_score', 0):.2f}")

        print(f"Has Contradiction: {assessment.get('has_contradiction')}")
        if assessment.get('missing_aspects'):
            print(f"Missing Aspects: {assessment.get('missing_aspects')}")
        print(f"\nReasoning:\n{assessment.get('reasoning', 'N/A')[:300]}...")
        
        state.update(result)
        return state
    
    # ========== Public Interface ==========
    
    def run(self, question: str, cpt_code: int = None, context: str = None) -> Dict[str, Any]:
        """
        运行简化的workflow
        
        Args:
            question: 用户问题
            cpt_code: 可选的CPT code（用于range filtering）
            context: 可选的上下文
            
        Returns:
            Dict: 包含完整的state信息
        """
        print("\n" + "🚀" + "="*78 + "🚀")
        print("Starting Simple Agentic RAG Workflow (No Iteration)")
        print("🚀" + "="*78 + "🚀")
        print(f"\nQuestion: {question}")
        if cpt_code:
            print(f"CPT Code: {cpt_code}")
        
        # 初始化state
        initial_state = AgenticRAGState(
            question=question,
            cpt_code=cpt_code,
            context=context,
        )
        
        # 运行graph
        final_state = self.graph.invoke(initial_state)
        
        print("\n" + "✅" + "="*78 + "✅")
        print("Workflow completed successfully!")
        print("✅" + "="*78 + "✅")
        
        # 保存到memory
        if self.enable_memory:
            try:
                saved_path = self.memory.save_execution(
                    question=question,
                    final_state=final_state,
                    workflow_type="simple",
                    mode=self.config.retrieval_mode,
                    success=True
                )
                print(f"\n💾 Workflow result saved to: {saved_path}")
            except Exception as e:
                print(f"\n⚠️  Failed to save memory: {e}")
        
        return final_state
    
    def visualize(self, output_path: str = "workflow_simple.png"):
        """
        可视化workflow graph
        
        Args:
            output_path: 输出图片路径
        """
        try:
            from IPython.display import Image, display
            display(Image(self.graph.get_graph().draw_mermaid_png()))
        except:
            print("Visualization requires IPython. Saving to file instead...")
            # 保存到文件
            graph_image = self.graph.get_graph().draw_mermaid_png()
            with open(output_path, "wb") as f:
                f.write(graph_image)
            print(f"Graph saved to {output_path}")
