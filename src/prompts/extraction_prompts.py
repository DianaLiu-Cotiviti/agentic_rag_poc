"""
Structured Extraction Agent prompts - versioned for output format iteration
"""

EXTRACTION_PROMPTS = {
    "v1": """Extract structured answer from the evidence chunks.

Question: {question}
Evidence: {evidence_summary}

Extract:
- Answer summary
- Allowed modifiers (if applicable)
- Rules and constraints
- Citations (chunk_id, page, section)

Output JSON:
{{
    "answer": "concise answer",
    "modifiers_allowed": ["51", "59", ...],
    "modifiers_not_allowed": [...],
    "rules": ["rule1", ...],
    "citations": [
        {{"chunk_id": "...", "page": 123, "section": "...", "quote": "..."}}
    ]
}}""",
    
    "v2": """You are a medical coding answer synthesizer.

QUESTION: {question}
EVIDENCE: {num_chunks} retrieved chunks

TASK: Synthesize a structured, citable answer.

OUTPUT SCHEMA:
{{
    "answer_summary": "1-2 sentence direct answer",
    
    "detailed_answer": {{
        "main_points": ["point1", "point2", ...],
        "modifiers": {{
            "allowed": ["51: Multiple Procedures", ...],
            "not_allowed": ["25: Significant, Separately Identifiable E/M", ...],
            "conditional": ["modifier: condition description", ...]
        }},
        "ptp_edits": {{
            "column1_code": "CPT1",
            "column2_code": "CPT2",
            "edit_indicator": "0|1|9",
            "explanation": "..."
        }},
        "rules": ["rule1", "rule2", ...],
        "constraints": ["constraint1", ...]
    }},
    
    "evidence_trace": [
        {{
            "chunk_id": "page_X_chunk_Y",
            "page_number": 123,
            "section": "Modifier Section",
            "relevance_score": 0.95,
            "quote": "exact text supporting the answer",
            "supports": "which part of answer this supports"
        }}
    ],
    
    "confidence_score": 0.0-1.0,
    "caveats": ["caveat1 if any", ...],
    
    "metadata": {{
        "retrieval_strategy": "hybrid",
        "num_chunks_used": 5,
        "coverage_assessment": "complete|partial|limited"
    }}
}}

CITATION RULES:
- Always quote exact text from chunks
- Include page numbers and sections
- Link each claim to specific evidence
- Mark low-confidence areas""",
}

DEFAULT_VERSION = "v2"
