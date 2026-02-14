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
   - **Use numbered citations [1] [2] [3] to support factual claims in your answer**
   - **Cite immediately after each specific factual statement, claim, or technical detail**
   - Example: "CPT code 14301 is an adjacent tissue transfer procedure [1]. It can be reported with modifier 59 when performed on a different anatomical site [2] [3]. However, NCCI edits restrict reporting it with CPT 27702 [4] [5]."
   - If multiple chunks support the same point, cite all relevant ones: [1] [2] [5]
   - **IMPORTANT: These 10 chunks were carefully selected as high-quality evidence - aim to extract all relevant information from them**
   - **For each chunk, check if it contains useful information for this question** - if yes, incorporate and cite it; if not, skip it
   - Most chunks should be relevant (aim for 7-10 citations), but don't force citations for truly irrelevant content
   - Transitional or general statements may not need citations, but all specific facts must be cited

3. **Generate Key Points - executive summary of your answer**
   - Key Points are **NOT sentence-by-sentence extraction** from your answer
   - They are a **high-level summary** highlighting the most important takeaways
   - Typically 3-5 concise bullet points covering different aspects
   - Each key point should include relevant citations [1] [2]
   - Example: "Modifier 59 allowed for distinct sites [2] [3]"
   - Think of them as "TL;DR" - what would a busy coder need to know quickly?

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

**CITATION GUIDELINES - READ CAREFULLY**:
1. **Extract all relevant information** - These 10 chunks were ranked highest by our retrieval system, so most should be useful
2. **Cite all specific factual claims** - when you state a fact, rule, code definition, or technical detail, add the citation [N]
3. **Cite multiple sources when relevant** - if several chunks support a claim, include all: [1] [2] [5]
4. **Natural citation placement** - add citations right after the fact they support
5. **Don't force irrelevant citations** - only cite chunks that actually answer the question (skip chunks if truly not relevant)
6. **Aim for comprehensive coverage** - if 7-10 chunks are relevant, use them all; don't stop at 4-5 out of habit

**Example format**:

```
Answer: "Based on the retrieved evidence, CPT code 14301 describes an adjacent tissue transfer procedure [1]. This involves moving tissue from one area to close a defect in an adjacent area [1] [2]. The code can be reported with modifier 59 when performed on a different anatomical site [2] [3]. However, NCCI edits restrict reporting it with CPT 27702 [4] [5]. The procedure is typically used for wound closure and reconstruction [6], and requires specific documentation of the defect size and tissue flap dimensions [7]. Medicare reimbursement guidelines specify particular clinical scenarios where this code applies [8], and providers must ensure proper documentation to support medical necessity [9] [10]."

Key Points (high-level summary of main takeaways):
- Adjacent tissue transfer procedure for wounds and defects [1] [6]
- Modifier 59 allowed for distinct anatomical sites [2] [3]
- NCCI bundling restrictions with certain codes [4] [5]
- Documentation of defect size and flap dimensions required [7] [8]
- Medical necessity and reimbursement guidelines must be followed [9] [10]

Citations: [
  {{"number": 1, "chunk_id": "chunk_000210"}},
  {{"number": 2, "chunk_id": "chunk_000345"}},
  {{"number": 3, "chunk_id": "chunk_000156"}},
  {{"number": 4, "chunk_id": "chunk_000892"}},
  {{"number": 5, "chunk_id": "chunk_000421"}},
  {{"number": 6, "chunk_id": "chunk_000567"}},
  {{"number": 7, "chunk_id": "chunk_000234"}},
  {{"number": 8, "chunk_id": "chunk_000678"}},
  {{"number": 9, "chunk_id": "chunk_000789"}},
  {{"number": 10, "chunk_id": "chunk_000921"}}
]

Confidence: 0.85 (strong evidence on main aspects, minor details missing)

Limitations: 
- Specific documentation requirements not fully detailed in evidence
- Payer-specific policies not covered
```
```

Generate your answer now."""
