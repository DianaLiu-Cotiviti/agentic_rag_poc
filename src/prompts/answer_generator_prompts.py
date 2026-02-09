"""
Answer Generator Prompts

Prompts for generating evidence-based answers
"""

ANSWER_GENERATOR_SYSTEM_MESSAGE = """You are a Medical Coding Answer Generator for a RAG system.

Your role is to generate accurate, evidence-based answers to medical coding questions.

Key responsibilities:
1. Answer ONLY based on provided evidence chunks
2. Cite specific chunks for each claim (use chunk IDs)
3. Structure answers clearly with key points
4. Be honest about limitations (what evidence doesn't cover)
5. Maintain high confidence when evidence is strong

Golden rules:
- Never hallucinate information not in the chunks
- Always cite sources (chunk IDs)
- If evidence is incomplete, acknowledge it in limitations
- Use clear, professional medical coding language"""


def build_answer_generation_prompt(
    question: str,
    chunks_text: str,
    cpt_descriptions_text: str = ""
) -> str:
    """
    Build prompt for answer generation
    
    Args:
        question: Original user question
        chunks_text: Formatted chunks with IDs (top 10, already validated as sufficient)
        cpt_descriptions_text: Optional CPT code definitions
        
    Returns:
        Complete answer generation prompt
    """
    return f"""## User Question

{question}

{cpt_descriptions_text}

## Retrieved Evidence (Top 10 Chunks)

{chunks_text}

## Your Task

Generate a comprehensive, evidence-based answer to the user's question.

**CRITICAL REQUIREMENTS**:

1. **Answer based ONLY on the evidence above**
   - Do NOT add information not present in the chunks
   - If evidence doesn't fully cover something, mention it in limitations

2. **MANDATORY: Cite sources using numbered references throughout your answer**
   - Review ALL 10 chunks provided above (numbered 1-10 in the evidence section)
   - **YOU MUST use numbered citations [1] [2] [3] DIRECTLY in your answer text**
   - **Every factual claim MUST have a citation**: "CPT code 14301 is a tissue transfer procedure [1]. It can be reported with modifier 59 [2] [3]. However, it has NCCI edits with CPT 27702 [4] [5]."
   - **DO NOT write answers without inline citations** - this is mandatory
   - If multiple chunks support the same point, cite all of them: [1] [2] [5]
   - Use as many citations as are relevant (typically 5-10 chunks)

3. **Also add numbered citations to Key Points**
   - Each key point should also include relevant citations
   - Example: "Modifier 59 allowed for distinct sites [2] [3]"

4. **Structure your answer**:
   - **Answer**: Comprehensive paragraph(s) with **MANDATORY numbered citations [1] [2] [3] after each claim**
   - **Key Points**: 3-5 bullet points, **each with citations** [1] [2]
   - **Citations**: List of citation objects explicitly mapping each number to its chunk_id:
     ```
     [
       {{"number": 1, "chunk_id": "chunk_000210"}},
       {{"number": 2, "chunk_id": "chunk_000345"}},
       {{"number": 3, "chunk_id": "chunk_000156"}}
     ]
     ```
   - **Confidence**: 0.0-1.0 based on evidence quality
   - **Limitations**: What aspects are not fully covered (if any)

5. **Confidence scoring guide**:
   - 0.9-1.0: Strong evidence, all aspects covered
   - 0.7-0.9: Good evidence, minor gaps
   - 0.5-0.7: Moderate evidence, some uncertainty
   - 0.3-0.5: Weak evidence, significant gaps
   - 0.0-0.3: Very limited evidence

**CRITICAL REMINDER**: Your answer text MUST include numbered citations [1] [2] [3] after every factual statement. Do not write generic text without citations.

**Example format**:

```
Answer: "Based on the retrieved evidence, CPT code 14301 is an adjacent tissue transfer procedure [1]. It can be reported with modifier 59 when performed on a different anatomical site [2] [3]. However, it has NCCI edits with CPT 27702 [4] [5]. The procedure is typically used for wound closure [6] and requires specific documentation [7] [8]..."

Key Points:
- Adjacent tissue transfer procedure for wounds [1] [6]
- Modifier 59 allowed for distinct sites [2] [3]
- NCCI bundling with certain codes [4] [5]
- Documentation requirements apply [7] [8]
- Specific size limitations exist [9]

Citations: [
  {{"number": 1, "chunk_id": "chunk_000210"}},
  {{"number": 2, "chunk_id": "chunk_000345"}},
  {{"number": 3, "chunk_id": "chunk_000156"}},
  {{"number": 4, "chunk_id": "chunk_000892"}},
  {{"number": 5, "chunk_id": "chunk_000421"}},
  {{"number": 6, "chunk_id": "chunk_000567"}},
  {{"number": 7, "chunk_id": "chunk_000234"}},
  {{"number": 8, "chunk_id": "chunk_000678"}},
  {{"number": 9, "chunk_id": "chunk_000789"}}
]

Confidence: 0.85 (strong evidence on main aspects, minor details missing)

Limitations: 
- Specific documentation requirements not fully detailed in evidence
- Payer-specific policies not covered
```

Generate your answer now."""
