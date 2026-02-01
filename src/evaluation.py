"""
Evaluation framework for Agentic RAG system.
Support for automated testing and human evaluation.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class EvaluationExample:
    """Single evaluation example with ground truth"""
    question: str
    cpt_codes: List[str]
    ground_truth_answer: str
    ground_truth_modifiers: Optional[List[str]] = None
    ground_truth_citations: Optional[List[Dict]] = None
    difficulty: str = "medium"  # easy, medium, hard
    category: str = "modifier"  # modifier, PTP, guideline, definition
    

class AgenticRAGEvaluator:
    """
    Evaluate Agentic RAG system performance.
    
    Metrics:
    1. Accuracy: Answer correctness
    2. Citation Precision/Recall: Evidence quality
    3. Efficiency: Retrieval steps needed
    4. Latency: Response time
    """
    
    def __init__(self, test_set_path: Optional[str] = None):
        self.test_set: List[EvaluationExample] = []
        if test_set_path:
            self.load_test_set(test_set_path)
    
    def load_test_set(self, path: str):
        """Load test examples from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.test_set = [
            EvaluationExample(**example) for example in data
        ]
    
    def evaluate_answer(
        self,
        predicted: Dict[str, Any],
        ground_truth: EvaluationExample
    ) -> Dict[str, float]:
        """
        Evaluate a single answer.
        
        Returns:
            Dict with metric scores
        """
        metrics = {}
        
        # 1. Answer accuracy (simple token overlap for now)
        metrics["answer_accuracy"] = self._calculate_answer_accuracy(
            predicted.get("answer_summary", ""),
            ground_truth.ground_truth_answer
        )
        
        # 2. Modifier accuracy (exact match)
        if ground_truth.ground_truth_modifiers:
            metrics["modifier_precision"], metrics["modifier_recall"], metrics["modifier_f1"] = \
                self._calculate_set_metrics(
                    predicted.get("detailed_answer", {}).get("modifiers", {}).get("allowed", []),
                    ground_truth.ground_truth_modifiers
                )
        
        # 3. Citation quality
        if ground_truth.ground_truth_citations:
            metrics["citation_precision"], metrics["citation_recall"], metrics["citation_f1"] = \
                self._calculate_citation_metrics(
                    predicted.get("evidence_trace", []),
                    ground_truth.ground_truth_citations
                )
        
        return metrics
    
    def _calculate_answer_accuracy(self, predicted: str, ground_truth: str) -> float:
        """Calculate answer accuracy using token overlap (simple F1)"""
        pred_tokens = set(predicted.lower().split())
        gt_tokens = set(ground_truth.lower().split())
        
        if not gt_tokens:
            return 0.0
        
        overlap = pred_tokens & gt_tokens
        
        if not overlap:
            return 0.0
        
        precision = len(overlap) / len(pred_tokens) if pred_tokens else 0
        recall = len(overlap) / len(gt_tokens) if gt_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def _calculate_set_metrics(
        self,
        predicted: List[str],
        ground_truth: List[str]
    ) -> tuple[float, float, float]:
        """Calculate precision, recall, F1 for set comparison"""
        pred_set = set(predicted)
        gt_set = set(ground_truth)
        
        if not gt_set:
            return 0.0, 0.0, 0.0
        
        tp = len(pred_set & gt_set)
        
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(gt_set) if gt_set else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return precision, recall, f1
    
    def _calculate_citation_metrics(
        self,
        predicted: List[Dict],
        ground_truth: List[Dict]
    ) -> tuple[float, float, float]:
        """Calculate citation quality metrics"""
        # Extract page numbers from citations
        pred_pages = set(c.get("page_number") for c in predicted if c.get("page_number"))
        gt_pages = set(c.get("page_number") for c in ground_truth if c.get("page_number"))
        
        if not gt_pages:
            return 0.0, 0.0, 0.0
        
        tp = len(pred_pages & gt_pages)
        
        precision = tp / len(pred_pages) if pred_pages else 0.0
        recall = tp / len(gt_pages) if gt_pages else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return precision, recall, f1
    
    def batch_evaluate(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: Optional[List[EvaluationExample]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of predictions.
        
        Returns:
            Aggregated metrics and per-example results
        """
        if ground_truths is None:
            ground_truths = self.test_set
        
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions must match ground truths")
        
        all_metrics = []
        results = []
        
        for pred, gt in zip(predictions, ground_truths):
            metrics = self.evaluate_answer(pred, gt)
            all_metrics.append(metrics)
            
            results.append({
                "question": gt.question,
                "category": gt.category,
                "difficulty": gt.difficulty,
                "metrics": metrics
            })
        
        # Aggregate metrics
        aggregated = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if key in m]
                aggregated[f"{key}_mean"] = np.mean(values) if values else 0.0
                aggregated[f"{key}_std"] = np.std(values) if values else 0.0
        
        # Category-wise breakdown
        category_metrics = {}
        for category in set(gt.category for gt in ground_truths):
            cat_metrics = [
                m for r, m in zip(results, all_metrics)
                if r["category"] == category
            ]
            
            if cat_metrics:
                category_metrics[category] = {
                    "count": len(cat_metrics),
                    "answer_accuracy_mean": np.mean([m.get("answer_accuracy", 0) for m in cat_metrics])
                }
        
        return {
            "overall": aggregated,
            "by_category": category_metrics,
            "per_example": results
        }
    
    def create_test_set_template(self, output_path: str = "test_set_template.json"):
        """Create a template test set file"""
        template = [
            {
                "question": "What modifiers are allowed with CPT 14301?",
                "cpt_codes": ["14301"],
                "ground_truth_answer": "Modifiers 50, 51, 59, 76, 77, 78, 79, and 91 are allowed with CPT 14301.",
                "ground_truth_modifiers": ["50", "51", "59", "76", "77", "78", "79", "91"],
                "ground_truth_citations": [
                    {"page_number": 123, "section": "Modifier Section"}
                ],
                "difficulty": "medium",
                "category": "modifier"
            },
            {
                "question": "Can CPT 31622 and 31623 be billed together?",
                "cpt_codes": ["31622", "31623"],
                "ground_truth_answer": "CPT 31622 and 31623 have a PTP edit. 31622 is column 1 and 31623 is column 2. Modifier is allowed to override.",
                "ground_truth_modifiers": None,
                "ground_truth_citations": [
                    {"page_number": 456, "section": "PTP Edits"}
                ],
                "difficulty": "hard",
                "category": "PTP"
            }
        ]
        
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        return output_path
