"""
Query Refiner Prompts

Prompts for generating refined queries based on missing aspects
"""

QUERY_REFINER_SYSTEM_MESSAGE = """You are a Medical Coding Query Refinement Expert for a RAG system.

Your role is to analyze insufficient evidence and generate targeted refined queries.

Key responsibilities:
1. Analyze missing aspects from Evidence Judge assessment
2. Generate specific, targeted queries for each missing aspect
3. Avoid redundancy with previous queries
4. Use medical coding terminology precisely

Golden rules:
- Each refined query should address ONE specific missing aspect
- Be specific and actionable (not generic)
- Use CPT codes, modifiers, medical terms explicitly
- Avoid repeating previous failed query patterns"""


def build_query_refiner_prompt(
    original_question: str,
    missing_aspects: list,
    previous_queries: list,
    evidence_assessment: dict
) -> str:
    """
    Build prompt for query refinement
    
    Args:
        original_question: Original user question
        missing_aspects: List of missing aspects from Evidence Judge
        previous_queries: Queries that failed to retrieve sufficient evidence
        evidence_assessment: Full assessment from Evidence Judge
        
    Returns:
        Complete query refinement prompt
    """
    # Format missing aspects
    missing_text = "\n".join([f"   {i+1}. {aspect}" for i, aspect in enumerate(missing_aspects)])
    
    # Format previous queries
    prev_queries_text = "\n".join([f"   - {q}" for q in previous_queries])
    
    return f"""## Original Question

{original_question}

## Evidence Assessment

**Coverage Score**: {evidence_assessment.get('coverage_score', 0):.2f} / 1.0
**Specificity Score**: {evidence_assessment.get('specificity_score', 0):.2f} / 1.0
**Is Sufficient**: {evidence_assessment.get('is_sufficient', False)}

**Reasoning**:
{evidence_assessment.get('reasoning', 'N/A')}

## Missing Aspects (Critical Gaps)

{missing_text}

## Previous Queries (That Failed)

{prev_queries_text}

## Your Task

Generate refined, targeted queries to address the missing aspects above.

**Requirements**:

1. **Generate ONE refined query for EACH missing aspect**
   - Be specific and actionable
   - Target the exact missing information

2. **Avoid patterns from previous failed queries**
   - Don't repeat the same keywords/structure
   - Try different angles or synonyms

3. **Use medical coding terminology**
   - Include CPT codes, modifiers, NCCI terms explicitly
   - Be precise with medical jargon

4. **Query Structure Guidelines**:
   - For modifier questions: "CPT [code] modifier [XX] [specific scenario]"
   - For NCCI edits: "CPT [code] NCCI bundling [specific procedure/code]"
   - For guidelines: "CPT [code] [specific guideline aspect] requirements"
   - For documentation: "CPT [code] documentation requirements [specific detail]"

**Example Refinements**:

```
Missing Aspect: "modifier 59 compatibility"
❌ Bad: "Tell me about CPT 14301" (too generic, repeats previous)
✅ Good: "CPT 14301 modifier 59 distinct anatomical sites requirements"

Missing Aspect: "NCCI bundling rules"
❌ Bad: "CPT 14301 information" (vague)
✅ Good: "CPT 14301 NCCI edits with adjacent tissue transfer codes"

Missing Aspect: "documentation requirements"
❌ Bad: "CPT 14301 documentation" (too broad)
✅ Good: "CPT 14301 operative report documentation wound size measurements"
```

Generate your refined queries now. Be specific and targeted!"""
