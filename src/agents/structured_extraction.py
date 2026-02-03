"""
Structured Extraction Agent - 结构化提取器
负责从证据中提取结构化答案
"""

from pydantic import BaseModel, Field
from .base import BaseAgent
from ..state import AgenticRAGState


class StructuredAnswer(BaseModel):
    """结构化答案"""
    answer: str = Field(description="完整答案")
    confidence_score: float = Field(ge=0, le=1, description="置信度")
    citations: list[dict] = Field(description="引用来源")
    reasoning: str = Field(description="推理过程")


class StructuredExtractionAgent(BaseAgent):
    """
    Structured Extraction Agent
    
    职责:
    1. 基于证据生成完整答案
    2. 提供引用来源
    3. 计算置信度
    4. 解释推理过程
    """
    
    def process(self, state: AgenticRAGState) -> dict:
        """
        提取结构化答案
        
        Args:
            state: 包含question, retrieved_chunks的状态
            
        Returns:
            dict: 包含structured_answer
        """
        question = state["question"]
        chunks = state.get("retrieved_chunks", [])
        
        # 构建prompt
        prompt = self._build_prompt(question, chunks)
        
        # 调用LLM提取答案
        response = self.client.beta.chat.completions.parse(
            model=self.config.azure_deployment_name,
            messages=[
                {"role": "system", "content": "你是医疗编码专家，基于证据提供准确答案。"},
                {"role": "user", "content": prompt}
            ],
            response_format=StructuredAnswer,
            temperature=self.config.agent_temperature
        )
        
        answer = response.choices[0].message.parsed
        
        return {
            "structured_answer": {
                "answer": answer.answer,
                "confidence_score": answer.confidence_score,
                "citations": answer.citations,
                "reasoning": answer.reasoning
            }
        }
    
    def _build_prompt(self, question: str, chunks: list) -> str:
        """构建Structured Extraction的prompt"""
        evidence = "\n\n".join([
            f"[文档{i+1}] {chunk.get('text', '')}"
            for i, chunk in enumerate(chunks)
        ])
        
        return f"""基于以下证据回答问题：

问题: {question}

证据:
{evidence}

请提供:
1. 完整准确的答案
2. 置信度分数 (0-1)
3. 引用的文档来源
4. 你的推理过程
"""
