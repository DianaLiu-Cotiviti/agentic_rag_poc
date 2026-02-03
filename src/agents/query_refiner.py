"""
Query Refiner Agent - 查询优化器
负责根据证据缺失优化查询
"""

from pydantic import BaseModel
from .base import BaseAgent
from ..state import AgenticRAGState


class RefinedQueries(BaseModel):
    """优化后的查询"""
    refined_queries: list[str]
    refinement_strategy: str
    reasoning: str


class QueryRefinerAgent(BaseAgent):
    """
    Query Refiner Agent
    
    职责:
    1. 分析证据评估的缺失方面
    2. 生成针对性的新查询
    3. 优化查询策略
    """
    
    def process(self, state: AgenticRAGState) -> dict:
        """
        优化查询
        
        Args:
            state: 包含question, evidence_assessment的状态
            
        Returns:
            dict: 包含refined_queries
        """
        question = state["question"]
        assessment = state.get("evidence_assessment", {})
        missing_aspects = assessment.get("missing_aspects", [])
        
        # 构建prompt
        prompt = self._build_prompt(question, missing_aspects)
        
        # 调用LLM生成优化查询
        response = self.client.beta.chat.completions.parse(
            model=self.config.azure_deployment_name,
            messages=[
                {"role": "system", "content": "你是查询优化专家。"},
                {"role": "user", "content": prompt}
            ],
            response_format=RefinedQueries,
            temperature=self.config.agent_temperature
        )
        
        refined = response.choices[0].message.parsed
        
        return {
            "refined_queries": refined.refined_queries
        }
    
    def _build_prompt(self, question: str, missing_aspects: list[str]) -> str:
        """构建Query Refiner的prompt"""
        missing_text = "\n".join([f"- {aspect}" for aspect in missing_aspects])
        
        return f"""根据证据缺失优化查询：

原始问题: {question}

缺失的方面:
{missing_text}

请生成2-3个针对性的新查询，专门针对缺失的方面。
同时说明你的优化策略。
"""
