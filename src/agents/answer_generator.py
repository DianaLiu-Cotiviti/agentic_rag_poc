"""
Answer Generator Agent - 答案生成器

基于 Evidence Judge 判定为 sufficient 的 top 10 chunks 生成最终答案。

核心职责:
1. 接收 original question + top 10 high-quality chunks
2. 生成结构化的、有证据支持的答案
3. 引用具体的 chunk 来源
4. 确保答案准确、完整、可追溯
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from .base import BaseAgent
from ..state import AgenticRAGState, RetrievalResult


class CitedAnswer(BaseModel):
    """
    Answer Generator 的输出结构
    
    包含答案文本和证据引用
    """
    answer: str = Field(
        description="Comprehensive answer to the user's question"
    )
    key_points: List[str] = Field(
        default_factory=list,
        description="Key points extracted from evidence (bullet points)"
    )
    citations: List[str] = Field(
        default_factory=list,
        description="Chunk IDs used as evidence (e.g., ['chunk_123', 'chunk_456'])"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score based on evidence quality (0.0-1.0)"
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Any limitations or caveats in the answer"
    )


class AnswerGeneratorAgent(BaseAgent):
    """
    Answer Generator Agent - 生成最终答案
    
    工作流程:
    1. 接收 original question 和 top 10 chunks（已被 Evidence Judge 验证为 sufficient）
    2. 构建 prompt，要求 LLM 基于这些 chunks 生成答案
    3. 返回结构化答案（包含 citations, key_points, confidence）
    
    设计原则:
    - 答案必须基于提供的 chunks（不能幻觉）
    - 必须引用具体的 chunk（可追溯性）
    - 如果证据不足某些方面，明确说明（limitations）
    """
    
    def __init__(self, config, client=None):
        """
        Args:
            config: Configuration object with Azure OpenAI settings
            client: Azure OpenAI client (optional, will use config.client if not provided)
        """
        self.config = config
        self._client = client if client is not None else getattr(config, 'client', None)
    
    @property
    def client(self):
        """Lazy initialization of LLM client"""
        if self._client is None:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                api_key=self.config.azure_openai_api_key,
                api_version=self.config.azure_api_version,
                azure_endpoint=self.config.azure_openai_endpoint
            )
        return self._client
    
    def process(self, state: AgenticRAGState) -> dict:
        """
        生成基于证据的答案
        
        Args:
            state: Contains question, retrieved_chunks (top 10), evidence_assessment
            
        Returns:
            dict: Contains final_answer
        """
        question = state["question"]
        chunks = state.get("retrieved_chunks", [])
        evidence_assessment = state.get("evidence_assessment", {})
        cpt_descriptions = state.get("cpt_descriptions", {})
        
        # 安全检查：确保有 chunks
        if not chunks:
            return {
                "final_answer": {
                    "answer": "无法生成答案：没有检索到相关证据。",
                    "key_points": [],
                    "citations": [],
                    "confidence": 0.0,
                    "limitations": ["未检索到任何相关文档"]
                }
            }
        
        # 构建 answer generation prompt
        prompt = self._build_answer_prompt(
            question=question,
            chunks=chunks,
            evidence_assessment=evidence_assessment,
            cpt_descriptions=cpt_descriptions
        )
        
        # 调用 LLM 生成结构化答案
        response = self.client.beta.chat.completions.parse(
            model=self.config.azure_deployment_name,
            messages=[
                {"role": "system", "content": self._get_system_message()},
                {"role": "user", "content": prompt}
            ],
            response_format=CitedAnswer,
            temperature=self.config.agent_temperature
        )
        
        answer = response.choices[0].message.parsed
        
        return {
            "final_answer": {
                "answer": answer.answer,
                "key_points": answer.key_points,
                "citations": answer.citations,
                "confidence": answer.confidence,
                "limitations": answer.limitations
            }
        }
    
    def _get_system_message(self) -> str:
        """System message for Answer Generator"""
        return """You are a Medical Coding Answer Generator for a RAG system.

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
    
    def _build_answer_prompt(
        self,
        question: str,
        chunks: List[RetrievalResult],
        evidence_assessment: dict,
        cpt_descriptions: dict = None
    ) -> str:
        """
        构建答案生成的 prompt
        
        Args:
            question: Original user question
            chunks: Top 10 chunks (already validated as sufficient by Evidence Judge)
            evidence_assessment: Evidence quality scores
            cpt_descriptions: CPT code definitions (if available)
        """
        # Format chunks with IDs
        chunks_text = self._format_chunks_with_ids(chunks)
        
        # Format CPT descriptions (if available)
        cpt_desc_text = ""
        if cpt_descriptions:
            cpt_desc_text = "\n\n### 📋 CPT Code Definitions\n\n"
            for code, description in cpt_descriptions.items():
                cpt_desc_text += f"**CPT {code}**: {description}\n\n"
        
        # Format evidence quality info
        coverage = evidence_assessment.get('coverage_score', 0.0)
        specificity = evidence_assessment.get('specificity_score', 0.0)
        
        prompt = f"""## User Question

{question}

## Evidence Quality Assessment

- **Coverage Score**: {coverage:.2f}/1.0
- **Specificity Score**: {specificity:.2f}/1.0
- **Evidence Status**: SUFFICIENT ✅

{cpt_desc_text}

## Retrieved Evidence (Top 10 Chunks)

{chunks_text}

## Your Task

Generate a comprehensive, evidence-based answer to the user's question.

**Requirements**:

1. **Answer based ONLY on the evidence above**
   - Do NOT add information not present in the chunks
   - If evidence doesn't fully cover something, mention it in limitations

2. **Cite your sources**
   - Use chunk IDs in your answer (e.g., "According to [chunk_123]...")
   - Include ALL chunk IDs you reference in the `citations` field

3. **Structure your answer**:
   - **Answer**: Comprehensive paragraph(s) answering the question
   - **Key Points**: 3-5 bullet points of main takeaways
   - **Citations**: List of chunk IDs used
   - **Confidence**: 0.0-1.0 based on evidence quality
   - **Limitations**: What aspects are not fully covered (if any)

4. **Confidence scoring guide**:
   - 0.9-1.0: Strong evidence, all aspects covered
   - 0.7-0.9: Good evidence, minor gaps
   - 0.5-0.7: Moderate evidence, some uncertainty
   - 0.3-0.5: Weak evidence, significant gaps
   - 0.0-0.3: Very limited evidence

**Example format**:

```
Answer: "Based on the retrieved evidence, CPT code 14301 is an adjacent tissue transfer procedure [chunk_abc]. It can be reported with modifier 59 when performed on a different anatomical site [chunk_def]. However, it has NCCI edits with CPT 27702 [chunk_ghi]..."

Key Points:
- Adjacent tissue transfer procedure for wounds
- Modifier 59 allowed for distinct sites
- NCCI bundling with certain codes
- Documentation requirements apply

Citations: [chunk_abc, chunk_def, chunk_ghi]

Confidence: 0.85 (strong evidence on main aspects, minor details missing)

Limitations: 
- Specific documentation requirements not detailed in evidence
- Payer-specific policies not covered
```

Generate your answer now."""
        
        return prompt
    
    def _format_chunks_with_ids(self, chunks: List[RetrievalResult]) -> str:
        """
        格式化 chunks，添加明确的 ID 标识
        
        Args:
            chunks: List of RetrievalResult objects
        """
        if not chunks:
            return "No chunks available."
        
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            # Extract chunk data
            if isinstance(chunk, dict):
                chunk_id = chunk.get("chunk_id", f"chunk_{i}")
                text = chunk.get("text", "")
                score = chunk.get("score", 0.0)
                metadata = chunk.get("metadata", {})
            else:
                chunk_id = chunk.chunk_id
                text = chunk.text
                score = chunk.score
                metadata = chunk.metadata
            
            # Format metadata
            cpt_info = f" [CPT: {metadata.get('cpt_code')}]" if metadata.get('cpt_code') else ""
            
            # Format chunk
            formatted.append(
                f"**[{chunk_id}]** (Score: {score:.4f}){cpt_info}\n"
                f"{text}\n"
            )
        
        return "\n---\n\n".join(formatted)
