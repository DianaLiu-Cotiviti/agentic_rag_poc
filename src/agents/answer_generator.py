"""
Answer Generator Agent - 答案生成器

基于 Evidence Judge 判定为 sufficient 的 top 10 chunks 生成最终答案。

核心职责:
1. 接收 original question + top 10 high-quality chunks
2. 生成结构化的、有证据支持的答案
3. 引用具体的 chunk 来源
4. 确保答案准确、完整、可追溯

设计原则:
- 答案必须基于提供的 chunks（不能幻觉）
- 必须引用具体的 chunk（可追溯性）
- 如果证据不足某些方面，明确说明（limitations）
- 代码简洁：prompts 在 prompts/，formatting 在 utils/
"""

from typing import Dict, Any
from pydantic import BaseModel, Field
from typing import Literal, List, Dict
from .base import BaseAgent
from ..state import AgenticRAGState
from ..prompts.answer_generator_prompts import (
    ANSWER_GENERATOR_SYSTEM_MESSAGE,
    build_answer_generation_prompt
)
from ..utils.chunk_formatting import (
    format_chunks_with_ids,
    format_cpt_descriptions
)


class Citation(BaseModel):
    """Single citation mapping citation number to chunk_id"""
    number: int = Field(description="Citation number used in answer, e.g., 1 for [1]")
    chunk_id: str = Field(description="Chunk ID being cited, e.g., 'chunk_000210'")


class CitedAnswer(BaseModel):
    """
    Answer Generator 的输出结构
    
    包含答案文本和证据引用（使用数字引用格式 [1] [2] [3]）
    
    注意：citation_map 不在此模型中，会在process()中自动生成
    """
    answer: str = Field(
        description="Comprehensive answer with inline numbered citations [1] [2] [3]. MUST include citations after each claim."
    )
    key_points: List[str] = Field(
        default_factory=list,
        description="Key points with numbered citations [1] [2], e.g., 'Modifier 59 allowed [2] [3]'"
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of citations mapping citation numbers to chunk IDs. E.g., [{number: 1, chunk_id: 'chunk_000210'}, {number: 2, chunk_id: 'chunk_000345'}]"
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
    2. 使用 prompts/answer_generator_prompts.py 中的 prompt
    3. 返回结构化答案（包含 citations, key_points, confidence）
    
    Token 优化策略:
    - 只接收已验证为 sufficient 的 top 10 chunks（通过 conditional edge）
    - 不重复展示 evidence_assessment（已在 Evidence Judge 完成）
    - CPT descriptions 单独格式化，避免冗余
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
        
        注意: 此方法只在 evidence is_sufficient=True 时被调用（通过 conditional edge）
        因此无需再检查 evidence_assessment 的质量分数
        
        Args:
            state: Contains question, retrieved_chunks (top 10), cpt_descriptions
            
        Returns:
            dict: Contains final_answer
        """
        question = state["question"]
        chunks = state.get("retrieved_chunks", [])
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
        
        # 使用 utils 格式化 chunks 和 CPT descriptions
        chunks_text = format_chunks_with_ids(chunks)
        cpt_desc_text = format_cpt_descriptions(cpt_descriptions)
        
        # 使用 prompts/ 中的 prompt builder
        prompt = build_answer_generation_prompt(
            question=question,
            chunks_text=chunks_text,
            cpt_descriptions_text=cpt_desc_text
        )
        
        # 调用 LLM 生成结构化答案
        response = self.client.beta.chat.completions.parse(
            model=self.config.azure_deployment_name,
            messages=[
                {"role": "system", "content": ANSWER_GENERATOR_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            response_format=CitedAnswer,
            temperature=self.config.agent_temperature
        )
        
        answer = response.choices[0].message.parsed
        
        # Generate citation_map from LLM's explicit Citation objects
        # This ensures correct mapping between citation numbers and chunk IDs
        citation_map = {
            citation.number: citation.chunk_id 
            for citation in answer.citations
        }
        
        # ✅ VALIDATION: Verify citation mapping integrity
        print(f"\n🔍 Citation Mapping Validation:")
        print(f"   LLM returned {len(answer.citations)} citation objects")
        
        # Display each citation mapping for verification
        for citation in sorted(answer.citations, key=lambda c: c.number):
            print(f"   [{citation.number}] → {citation.chunk_id}")
        
        # Check for duplicate citation numbers
        citation_numbers = [c.number for c in answer.citations]
        if len(citation_numbers) != len(set(citation_numbers)):
            duplicates = [n for n in citation_numbers if citation_numbers.count(n) > 1]
            print(f"   ⚠️  WARNING: Duplicate citation numbers detected: {set(duplicates)}")
        else:
            print(f"   ✅ All citation numbers are unique")
        
        # Verify citation_map matches LLM output
        print(f"   ✅ Generated citation_map with {len(citation_map)} entries")
        
        return {
            "final_answer": {
                "answer": answer.answer,
                "key_points": answer.key_points,
                "citation_map": citation_map,  # Explicit mapping from LLM
                "confidence": answer.confidence,
                "limitations": answer.limitations
            }
        }

