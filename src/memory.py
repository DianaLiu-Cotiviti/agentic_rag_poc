"""
Workflow Memory Management
==========================

Save and manage workflow execution history for:
- Debugging and tracing
- Performance analysis
- Long-term learning
- User session history
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class WorkflowMemory:
    """
    Workflow execution record manager
    
    Features:
    - Save complete state for each workflow execution
    - Maintain historical records (with timestamps)
    - Provide latest.json for quick access to the latest results
    
    Usage:
        memory = WorkflowMemory(memory_dir="memory")
        memory.save_execution(
            question="What is CPT 14301?",
            final_state=state,
            workflow_type="simple"
        )
    """
    
    def __init__(self, memory_dir: str = "memory"):
        """
        Initialize Memory manager
        
        Args:
            memory_dir: Storage directory path (relative to project root)
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
        Save single workflow execution result
        
        Args:
            question: User question
            final_state: Final state of the workflow
            workflow_type: Workflow type ("simple", "with_retry", "full", etc.)
            mode: Retrieval mode ("direct", "planning", "tool_calling")
            success: Whether execution was successful
            error: Error message (if failed)
            
        Returns:
            Path: Path to the saved file
        """
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_part = f"_{mode}" if mode else ""
        filename = f"workflow_{workflow_type}{mode_part}_{timestamp}.json"
        filepath = self.memory_dir / filename
        
        # Build memory data
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
        
        # Save timestamped file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
        
        # Also save as latest.json
        mode_part = f"_{mode}" if mode else ""
        latest_path = self.memory_dir / f"latest_{workflow_type}{mode_part}.json"
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def _extract_orchestrator_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Orchestrator output"""
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
        """Extract Query Planner output"""
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
        """Extract Retrieval output (only save summary to avoid large files)"""
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
                for chunk in chunks[:10]  # Only save top 10
            ],
            "retrieval_metadata": state.get('retrieval_metadata', {})
        }
    
    def _extract_evidence_judge_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Evidence Judge output"""
        assessment = state.get('evidence_assessment', {})
        
        return {
            "is_sufficient": assessment.get('is_sufficient') if assessment else None,
            "coverage_score": assessment.get('coverage_score') if assessment else None,
            "specificity_score": assessment.get('specificity_score') if assessment else None,

            "has_contradiction": assessment.get('has_contradiction') if assessment else None,
            "missing_aspects": assessment.get('missing_aspects') if assessment else [],
            "reasoning_preview": (assessment.get('reasoning', '') if assessment else '')[:500],
            "retry_count": state.get('retry_count', 0)
        }
    
    def _extract_query_refiner_data(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract Query Refiner output (if exists)"""
        refined_queries = state.get('refined_queries')
        if refined_queries is None:
            return None
        
        return {
            "refined_queries": refined_queries
        }
    
    def _extract_structured_extraction_data(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract Structured Extraction output (if exists)"""
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
        """Safely get object attribute (supports both dict and object)"""
        if isinstance(obj, dict):
            return obj.get(attr, default)
        else:
            return getattr(obj, attr, default)
    
    def load_latest(self, workflow_type: str = "simple") -> Optional[Dict[str, Any]]:
        """
        Load the latest execution record
        
        Args:
            workflow_type: Workflow type
            
        Returns:
            Dict: Memory data, None if not exists
        """
        latest_path = self.memory_dir / f"latest_{workflow_type}.json"
        if not latest_path.exists():
            return None
        
        with open(latest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_history(self, workflow_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List execution history
        
        Args:
            workflow_type: Optional, only list specific workflow type
            limit: Return the latest N records
            
        Returns:
            List[Dict]: History list (sorted by time descending)
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
