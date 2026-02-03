"""
Evidence Judge Agent - 证据评判官
负责评估检索到的证据是否充分
"""

from pydantic import BaseModel, Field
from .base import BaseAgent
from ..state import AgenticRAGState


class EvidenceAssessment(BaseModel):
    """证据评估结果"""
    is_sufficient: bool = Field(description="证据是否充分")
    coverage_score: float = Field(ge=0, le=1, description="覆盖度分数")
    specificity_score: float = Field(ge=0, le=1, description="特定性分数")
    citation_count: int = Field(ge=0, description="引用数量")
    missing_aspects: list[str] = Field(description="缺失的方面")
    reasoning: str


class EvidenceJudgeAgent(BaseAgent):
    """
    Evidence Judge Agent
    
    职责:
    1. 评估检索证据的充分性
    2. 计算覆盖度和特定性分数
    3. 识别缺失的方面
    4. 决定是否需要重试
    """
    
    def process(self, state: AgenticRAGState) -> dict:
        """
        评估证据质量
        
        Args:
            state: 包含question, retrieved_chunks的状态
            
        Returns:
            dict: 包含evidence_assessment
        """
        question = state["question"]
        chunks = state.get("retrieved_chunks", [])
        
        # 如果没有检索到内容
        if not chunks:
            return {
                "evidence_assessment": {
                    "is_sufficient": False,
                    "coverage_score": 0.0,
                    "specificity_score": 0.0,
                    "citation_count": 0,
                    "missing_aspects": ["所有方面"],
                    "reasoning": "未检索到任何相关文档"
                }
            }
        
        # 构建prompt
        prompt = self._build_prompt(question, chunks)
        
        # 调用LLM评估
        response = self.client.beta.chat.completions.parse(
            model=self.config.azure_deployment_name,
            messages=[
                {"role": "system", "content": "你是证据评估专家。"},
                {"role": "user", "content": prompt}
            ],
            response_format=EvidenceAssessment,
            temperature=self.config.agent_temperature
        )
        
        assessment = response.choices[0].message.parsed
        
        return {
            "evidence_assessment": {
                "is_sufficient": assessment.is_sufficient,
                "coverage_score": assessment.coverage_score,
                "specificity_score": assessment.specificity_score,
                "citation_count": assessment.citation_count,
                "missing_aspects": assessment.missing_aspects,
                "reasoning": assessment.reasoning
            }
        }
    
    def _build_prompt(self, question: str, chunks: list) -> str:
        """构建Evidence Judge的prompt"""
        chunks_text = "\n\n".join([
            f"[文档{i+1}] {chunk.get('text', '')[:200]}..."
            for i, chunk in enumerate(chunks[:5])  # 只展示前5个
        ])
        
        return f"""评估以下检索证据是否足以回答问题：

问题: {question}

检索到的证据:
{chunks_text}

请评估:
1. 证据是否充分 (是/否)
2. 覆盖度分数 (0-1)
3. 特定性分数 (0-1)
4. 可用引用数量
5. 缺失哪些方面
6. 你的推理过程
"""
