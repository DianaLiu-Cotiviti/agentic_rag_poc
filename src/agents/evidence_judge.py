"""
Evidence Judge Agent - 证据评判官

负责评估检索到的证据质量，判断：
1. 证据是否充分回答问题 (is_sufficient)
2. 证据覆盖度 (coverage_score)
3. 证据相关性和准确性 (specificity_score)
4. 可引用的高质量chunk数量 (citation_count)
5. 是否存在矛盾信息 (has_contradiction)
6. 缺失的方面 (missing_aspects)
"""

from typing import List
from pydantic import BaseModel, Field
from .base import BaseAgent
from ..state import AgenticRAGState, RetrievalResult
from ..prompts.evidence_judge_prompts import (
    EVIDENCE_JUDGE_SYSTEM_MESSAGE,
    build_evidence_judgment_prompt
)


class EvidenceJudgment(BaseModel):
    """
    Evidence Judge的评估结果
    
    判断标准：
    - is_sufficient: 证据是否足够回答问题（综合考虑数量、质量、覆盖度）
    - coverage_score: 证据对问题各方面的覆盖程度（0.0-1.0）
    - specificity_score: 证据的特定性和准确性（0.0-1.0）
    - citation_count: 可引用的高质量chunk数量（score > threshold）
    - has_contradiction: 检索结果中是否存在矛盾信息
    - missing_aspects: 问题中未被覆盖的方面（用于指导重试）
    - reasoning: 评估推理过程（解释为什么sufficient/insufficient）
    """
    is_sufficient: bool = Field(
        description="Whether the evidence is sufficient to answer the question"
    )
    coverage_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How well the evidence covers different aspects of the question (0.0-1.0)"
    )
    specificity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How specific and accurate the evidence is (0.0-1.0)"
    )
    citation_count: int = Field(
        ge=0,
        description="Number of high-quality chunks that can be cited"
    )
    has_contradiction: bool = Field(
        description="Whether there are contradictory statements in the evidence"
    )
    missing_aspects: List[str] = Field(
        default_factory=list,
        description="Aspects of the question not covered by current evidence"
    )
    reasoning: str = Field(
        description="Detailed reasoning for the sufficiency judgment"
    )


class EvidenceJudgeAgent(BaseAgent):
    """
    Evidence Judge Agent - 评估检索证据质量
    
    核心职责：
    1. 判断证据是否充分 (is_sufficient)
       - 考虑问题类型（简单CPT lookup vs 复杂billing规则）
       - 考虑证据数量和质量
       - 考虑覆盖度
    
    2. 评估证据质量指标：
       - coverage_score: 覆盖问题的多个方面（CPT code定义、modifier、bundling等）
       - specificity_score: 证据的准确性和相关性
       - citation_count: 可用于引用的高质量chunks
    
    3. 识别问题：
       - has_contradiction: 检测矛盾信息
       - missing_aspects: 识别缺失的方面
    
    4. 指导下一步行动：
       - 如果insufficient，missing_aspects指导query refinement
       - 如果sufficient，高质量chunks用于answer generation
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
        评估检索证据质量
        
        Args:
            state: Contains question, question_type, retrieved_chunks, query_candidates
            
        Returns:
            dict: Contains evidence_assessment
        """
        question = state["question"]
        question_type = state.get("question_type", "general")
        chunks = state.get("retrieved_chunks", [])
        retrieval_metadata = state.get("retrieval_metadata", {})
        
        # 如果没有检索到内容 - 明确insufficient
        if not chunks:
            return {
                "evidence_assessment": {
                    "is_sufficient": False,
                    "coverage_score": 0.0,
                    "specificity_score": 0.0,
                    "citation_count": 0,
                    "has_contradiction": False,
                    "missing_aspects": ["No chunks retrieved - all aspects missing"],
                    "reasoning": "No relevant chunks were retrieved. Need to refine query or adjust retrieval strategy."
                }
            }
        
        # 构建prompt用于LLM评估
        # 注意：只用 original question 和 retrieved chunks 评估
        # 不需要 sub-queries（它们只是检索手段，不是评估目标）
        prompt = self._build_judgment_prompt(
            question=question,
            question_type=question_type,
            chunks=chunks,
            retrieval_metadata=retrieval_metadata
        )
        
        # 调用LLM进行结构化评估
        response = self.client.beta.chat.completions.parse(
            model=self.config.azure_deployment_name,
            messages=[
                {"role": "system", "content": EVIDENCE_JUDGE_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            response_format=EvidenceJudgment,
            temperature=self.config.agent_temperature
        )
        
        judgment = response.choices[0].message.parsed
        
        return {
            "evidence_assessment": {
                "is_sufficient": judgment.is_sufficient,
                "coverage_score": judgment.coverage_score,
                "specificity_score": judgment.specificity_score,
                "citation_count": judgment.citation_count,
                "has_contradiction": judgment.has_contradiction,
                "missing_aspects": judgment.missing_aspects,
                "reasoning": judgment.reasoning
            }
        }
    
    def _build_judgment_prompt(
        self,
        question: str,
        question_type: str,
        chunks: List[RetrievalResult],
        retrieval_metadata: dict
    ) -> str:
        """
        构建Evidence Judge的评估prompt
        
        评估逻辑：
        - 评估目标：original question（不是sub-queries）
        - 评估证据：retrieved chunks（已融合）
        - 评估标准：question_type 对应的 required aspects
        
        Args:
            question: Original user question（评估目标）
            question_type: Question type
            chunks: Retrieved chunks（已融合的15-20个chunks）
            retrieval_metadata: Retrieval metadata
        """
        # Format chunks
        chunks_text = self._format_chunks_for_evaluation(chunks)
        
        # Extract metadata
        retrieval_mode = retrieval_metadata.get("mode", "unknown")
        strategies_used = retrieval_metadata.get("strategies_used", "N/A")
        
        # Use centralized prompt builder
        return build_evidence_judgment_prompt(
            question=question,
            question_type=question_type,
            chunks_text=chunks_text,
            retrieval_mode=retrieval_mode,
            strategies_used=str(strategies_used),
            total_chunks=len(chunks)
        )
    
    def _format_chunks_for_evaluation(self, chunks: List[RetrievalResult], max_chunks: int = 10) -> str:
        """
        Format chunks for LLM evaluation
        
        Args:
            chunks: List of RetrievalResult objects
            max_chunks: Maximum number of chunks to show (default: 10)
        """
        if not chunks:
            return "No chunks retrieved."
        
        formatted_chunks = []
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            # Handle both RetrievalResult objects and dicts
            if isinstance(chunk, dict):
                chunk_id = chunk.get("chunk_id", "unknown")
                score = chunk.get("score", 0.0)
                text = chunk.get("text", "")
                metadata = chunk.get("metadata", {})
            else:
                chunk_id = chunk.chunk_id
                score = chunk.score
                text = chunk.text
                metadata = chunk.metadata
            
            # Get CPT code from metadata if available
            cpt_info = ""
            if metadata.get("cpt_code"):
                cpt_info = f" [CPT: {metadata['cpt_code']}]"
            
            formatted_chunks.append(
                f"**Chunk {i}** (ID: {chunk_id}, Score: {score:.4f}){cpt_info}\n{text[:500]}..."
            )
        
        chunks_text = "\n\n".join(formatted_chunks)
        
        if len(chunks) > max_chunks:
            chunks_text += f"\n\n... and {len(chunks) - max_chunks} more chunks"
        
        return chunks_text

