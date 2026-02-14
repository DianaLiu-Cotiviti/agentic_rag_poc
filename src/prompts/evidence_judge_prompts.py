"""
Evidence Judge Prompts

Comprehensive prompts for Evidence Quality Assessment
"""


# System message for Evidence Judge
EVIDENCE_JUDGE_SYSTEM_MESSAGE = """You are an Evidence Quality Judge for a medical coding RAG system.

Your role is to critically evaluate whether retrieved evidence is SUFFICIENT and HIGH-QUALITY enough to answer the user's question.

Key responsibilities:
1. Assess if evidence covers all aspects of the question
2. Evaluate evidence quality (specificity, relevance, accuracy)
3. Identify contradictions or inconsistencies
4. Determine if more retrieval is needed
5. Provide actionable feedback for query refinement

Be thorough and critical - better to request more evidence than provide incomplete answers."""


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


def build_evidence_judgment_prompt(
    question: str,
    question_type: str,
    chunks_text: str,
    retrieval_mode: str,
    strategies_used: str,
    total_chunks: int
) -> str:
    """
    Build comprehensive evaluation prompt for Evidence Judge
    
    Evaluation Logic:
    - Evaluation Target: original question (user's original question)
    - Evaluation Evidence: retrieved chunks (15-20 high-scoring chunks after fusion)
    - Evaluation Method: Determine if required aspects are covered based on question_type
    
    Note: Sub-queries are NOT needed!
    - Sub-queries are only a retrieval method (used for Query Planner → Retrieval Router)
    - Evidence Judge evaluates: Can these chunks answer the original question?
    - NOT: Can these chunks answer each sub-query?
    
    Guidelines:
    - Coverage: Does evidence cover all required aspects for this question type?
    - Specificity: Is evidence specific to exact CPT codes/topics asked?
    - Contradiction: Any conflicting information?
    - Missing Aspects: What specific aspects are missing (for query refinement)?
    
    Args:
        question: Original user question (evaluation target)
        question_type: Question type (cpt_code_lookup, billing_compatibility, etc.)
        chunks_text: Formatted chunks for evaluation (chunks after fusion)
        retrieval_mode: Which retrieval mode was used
        strategies_used: Which strategies were used
        total_chunks: Total number of chunks retrieved
        
    Returns:
        Complete evaluation prompt
    """
    return f"""## Question to Answer

**Original Question**: {question}
**Question Type**: {question_type}

## Retrieval Information

**Retrieval Mode**: {retrieval_mode}
**Strategies Used**: {strategies_used}
**Total Chunks Retrieved**: {total_chunks}

## Retrieved Evidence

{chunks_text}

## Your Task

Critically evaluate whether this evidence is **SUFFICIENT** to answer the original question above.

**Evaluation Criteria**:

### 1. Coverage Score (0.0-1.0)

Does the evidence cover all aspects required for this question type?

**Required Aspects by Question Type**:

- **cpt_code_lookup**: 
  • CPT code definition
  • Anatomical location/scope
  • Procedure description
  • Typical usage scenarios
  • (Optional: Modifiers, bundling rules, documentation)

- **billing_compatibility**: 
  • Definition of BOTH CPT codes
  • NCCI edits for code pair
  • Explicit compatibility statement
  • Modifier requirements
  • (Optional: Documentation requirements, common scenarios)

- **modifier_query**: 
  • Modifier definition
  • When to use
  • Usage rules
  • Examples
  • (Optional: Contraindications, bundling implications)

- **concept_explanation**: 
  • Clear definition
  • Multiple examples
  • Practical application
  • (Optional: Edge cases, contraindications)

- **bundling_query**: 
  • NCCI rules for specific code pair
  • Edit type (0, 1, 9)
  • Modifier bypass rules
  • (Optional: Documentation requirements)

**Scoring Guide**:
- 1.0: All required aspects fully covered
- 0.8-0.9: Most required aspects covered
- 0.6-0.7: Core aspects covered, minor gaps
- 0.4-0.5: Partial coverage, significant gaps
- 0.0-0.3: Minimal coverage, mostly irrelevant

---

### 2. Specificity Score (0.0-1.0)

How specific and accurate is the evidence?

- **High (0.8-1.0)**: Exact CPT codes, concrete examples, precise details
- **Medium (0.5-0.7)**: Related but not exact, some generic info
- **Low (0.0-0.4)**: Generic, vague, no specific codes/examples

**Red Flags** (lower score):
- Mentions different CPT codes than asked
- Generic advice ("consult NCCI manual...")
- No concrete examples
- Information is outdated or ambiguous

---

### 3. Contradiction Detection

Check for conflicting statements:
- Different rules for same code
- Contradictory modifier requirements  
- Incompatible usage guidelines
- Conflicting numbers/values

If found: Set `has_contradiction = True` and explain in reasoning

---

### 4. Sufficiency Decision

**SUFFICIENT (True)** when ALL of:
- Coverage ≥ 0.7
- Specificity ≥ 0.7  
- No contradictions
- Can confidently answer the question

**INSUFFICIENT (False)** when ANY of:
- Coverage < 0.5 (missing critical aspects)
- Specificity < 0.5 (too generic)
- Contradictions detected
- Evidence is ambiguous or incomplete

---

### 5. Missing Aspects

If insufficient, list **SPECIFIC** missing aspects:

**Good examples** ✅:
- "NCCI bundling rules for CPT pair 14301+27702"
- "Modifier 59 usage requirements for this scenario"
- "Definition and usage of CPT 27702"

**Bad examples** ❌:
- "More information needed"
- "Better evidence required"

---

### 6. Reasoning

Explain your judgment:
- Why these coverage/specificity scores?
- Which chunks were high vs low quality?
- What's covered vs missing?
- How did you reach sufficiency decision?
- Describe any contradictions

**Example format**:
```
Coverage (0.X): [aspects covered and missing]
Specificity (0.X): [relevance evaluation]
Contradictions: [Yes/No - details if yes]
Decision: SUFFICIENT/INSUFFICIENT because [reason]
```

---

Provide your structured judgment now."""
