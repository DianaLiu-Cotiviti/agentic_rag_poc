"""
Workflow Memory Management
==========================

保存和管理workflow执行历史，用于：
- Debug和追溯
- 性能分析
- Long-term learning
- 用户会话历史
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class WorkflowMemory:
    """
    Workflow执行记录管理器
    
    功能：
    - 保存每次workflow执行的完整state
    - 维护历史记录（带时间戳）
    - 提供latest.json快速访问最新结果
    
    用法：
        memory = WorkflowMemory(memory_dir="memory")
        memory.save_execution(
            question="What is CPT 14301?",
            final_state=state,
            workflow_type="simple"
        )
    """
    
    def __init__(self, memory_dir: str = "memory"):
        """
        初始化Memory管理器
        
        Args:
            memory_dir: 存储目录路径（相对于项目根目录）
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
    
    def save_execution(
        self,
        question: str,
        final_state: Dict[str, Any],
        workflow_type: str = "simple",
        mode: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None
    ) -> Path:
        """
        保存单次workflow执行结果
        
        Args:
            question: 用户问题
            final_state: Workflow的最终state
            workflow_type: Workflow类型 ("simple", "with_retry", "full"等)
            mode: Retrieval mode ("direct", "planning", "tool_calling")
            success: 是否成功执行
            error: 错误信息（如果失败）
            
        Returns:
            Path: 保存的文件路径
        """
        # 生成时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_part = f"_{mode}" if mode else ""
        filename = f"workflow_{workflow_type}{mode_part}_{timestamp}.json"
        filepath = self.memory_dir / filename
        
        # 构建memory数据
        memory_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "workflow_type": workflow_type,
                "success": success,
                "error": error
            },
            
            # Step 1: Orchestrator outputs
            "orchestrator": self._extract_orchestrator_data(final_state),
            
            # Step 2: Query Planner outputs
            "query_planner": self._extract_query_planner_data(final_state),
            
            # Step 3: Retrieval outputs
            "retrieval": self._extract_retrieval_data(final_state),
            
            # Step 4: Evidence Judge outputs
            "evidence_judge": self._extract_evidence_judge_data(final_state),
            
            # Step 5: Query Refiner outputs (if exists)
            "query_refiner": self._extract_query_refiner_data(final_state),
            
            # Step 6: Structured Extraction outputs (if exists)
            "structured_extraction": self._extract_structured_extraction_data(final_state),
            
            # Complete state (for full context)
            "complete_state_keys": list(final_state.keys())
        }
        
        # 保存带时间戳的文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
        
        # 同时保存为latest.json
        mode_part = f"_{mode}" if mode else ""
        latest_path = self.memory_dir / f"latest_{workflow_type}{mode_part}.json"
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def _extract_orchestrator_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """提取Orchestrator的输出"""
        return {
            "question_type": state.get('question_type'),
            "question_keywords": state.get('question_keywords'),
            "question_complexity": state.get('question_complexity'),
            "retrieval_strategies": state.get('retrieval_strategies'),
            "enable_retry": state.get('enable_retry'),
            "max_retry_allowed": state.get('max_retry_allowed'),
            "require_structured_output": state.get('require_structured_output'),
            "reasoning": state.get('orchestrator_reasoning')
        }
    
    def _extract_query_planner_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """提取Query Planner的输出"""
        query_candidates = state.get('query_candidates', [])
        
        return {
            "num_candidates": len(query_candidates),
            "query_candidates": [
                {
                    "query": self._get_attr(qc, 'query'),
                    "query_type": self._get_attr(qc, 'query_type'),
                    "weight": self._get_attr(qc, 'weight')
                }
                for qc in query_candidates
            ],
            "messages": state.get('messages', [])
        }
    
    def _extract_retrieval_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """提取Retrieval的输出（只保存摘要，避免文件过大）"""
        chunks = state.get('retrieved_chunks', [])
        
        return {
            "num_chunks": len(chunks),
            "top_10_chunks_summary": [
                {
                    "chunk_id": self._get_attr(chunk, 'chunk_id'),
                    "score": self._get_attr(chunk, 'score'),
                    "text_preview": str(self._get_attr(chunk, 'text'))[:200] + "...",
                    "metadata": self._get_attr(chunk, 'metadata')
                }
                for chunk in chunks[:10]  # 只保存前10个
            ],
            "retrieval_metadata": state.get('retrieval_metadata', {})
        }
    
    def _extract_evidence_judge_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """提取Evidence Judge的输出"""
        assessment = state.get('evidence_assessment', {})
        
        return {
            "is_sufficient": assessment.get('is_sufficient') if assessment else None,
            "coverage_score": assessment.get('coverage_score') if assessment else None,
            "specificity_score": assessment.get('specificity_score') if assessment else None,
            "citation_count": assessment.get('citation_count') if assessment else None,
            "has_contradiction": assessment.get('has_contradiction') if assessment else None,
            "missing_aspects": assessment.get('missing_aspects') if assessment else [],
            "reasoning_preview": (assessment.get('reasoning', '') if assessment else '')[:500],
            "retry_count": state.get('retry_count', 0)
        }
    
    def _extract_query_refiner_data(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """提取Query Refiner的输出（如果有）"""
        refined_queries = state.get('refined_queries')
        if refined_queries is None:
            return None
        
        return {
            "refined_queries": refined_queries
        }
    
    def _extract_structured_extraction_data(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """提取Structured Extraction的输出（如果有）"""
        answer = state.get('structured_answer')
        if answer is None:
            return None
        
        return {
            "answer_preview": str(self._get_attr(answer, 'answer'))[:500] + "...",
            "rules_count": len(self._get_attr(answer, 'rules', [])),
            "modifiers_count": len(self._get_attr(answer, 'allowed_modifiers', [])),
            "constraints_count": len(self._get_attr(answer, 'constraints', [])),
            "confidence": self._get_attr(answer, 'confidence')
        }
    
    def _get_attr(self, obj, attr: str, default=None):
        """安全获取对象属性（支持dict和object）"""
        if isinstance(obj, dict):
            return obj.get(attr, default)
        else:
            return getattr(obj, attr, default)
    
    def load_latest(self, workflow_type: str = "simple") -> Optional[Dict[str, Any]]:
        """
        加载最新的执行记录
        
        Args:
            workflow_type: Workflow类型
            
        Returns:
            Dict: Memory数据，如果不存在返回None
        """
        latest_path = self.memory_dir / f"latest_{workflow_type}.json"
        if not latest_path.exists():
            return None
        
        with open(latest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_history(self, workflow_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        列出历史记录
        
        Args:
            workflow_type: 可选，只列出特定类型的workflow
            limit: 返回最近N条记录
            
        Returns:
            List[Dict]: 历史记录列表（按时间倒序）
        """
        pattern = f"workflow_{workflow_type}_*.json" if workflow_type else "workflow_*.json"
        files = sorted(
            self.memory_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:limit]
        
        history = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                history.append({
                    "filename": file.name,
                    "timestamp": data['metadata']['timestamp'],
                    "question": data['metadata']['question'],
                    "success": data['metadata'].get('success', True)
                })
        
        return history
