"""
Evidence Judge Agent prompts - versioned for threshold tuning
"""

EVIDENCE_JUDGE_PROMPTS = {
    "v1": """Evaluate if retrieved evidence is sufficient to answer the question.

Question: {question}
Retrieved chunks: {num_chunks} chunks

Criteria:
1. Coverage: Does evidence address all aspects?
2. Specificity: Is it specific to the CPT codes asked?
3. Consistency: Any contradictions?

Output JSON:
{{
    "is_sufficient": true/false,
    "confidence": 0.0-1.0,
    "missing_aspects": ["what's missing", ...],
    "reasoning": "explanation"
}}""",
    
    "v2": """You are an evidence quality assessor for medical coding queries.

QUESTION: {question}
RETRIEVED: {num_chunks} chunks from {sources}

EVALUATION CRITERIA:
1. COVERAGE (0-1): Do chunks cover all question aspects?
   - All CPT codes mentioned? ✓/✗
   - All modifiers addressed? ✓/✗
   - Sufficient context? ✓/✗

2. SPECIFICITY (0-1): Is evidence directly relevant?
   - Exact CPT code match vs general guidelines
   - Modifier-specific vs general policy

3. CONSISTENCY (0-1): Any conflicts or ambiguity?
   - Multiple chunks saying different things?
   - Version conflicts (e.g., 2023 vs 2024)?

4. CITATION QUALITY (0-1): Can we cite evidence?
   - Page numbers available?
   - Section headers clear?

DECISION THRESHOLDS:
- Sufficient: coverage ≥ 0.7 AND specificity ≥ 0.6 AND consistency ≥ 0.8
- Retry needed: coverage < 0.5 OR specificity < 0.4

Output JSON:
{{
    "is_sufficient": true/false,
    "scores": {{
        "coverage": 0.0-1.0,
        "specificity": 0.0-1.0,
        "consistency": 0.0-1.0,
        "citation_quality": 0.0-1.0
    }},
    "overall_confidence": 0.0-1.0,
    "missing_aspects": ["..."],
    "next_action": "answer|retry|refine_query",
    "reasoning": "..."
}}""",
}

DEFAULT_VERSION = "v2"

# Configurable thresholds for RL tuning
THRESHOLDS = {
    "v1": {
        "min_confidence": 0.6,
    },
    "v2": {
        "min_coverage": 0.7,
        "min_specificity": 0.6,
        "min_consistency": 0.8,
        "min_overall": 0.65,
    }
}
