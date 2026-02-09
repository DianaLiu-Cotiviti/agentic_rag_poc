"""
Evidence Judge Agent - 证据评判官

负责评估检索到的证据质量，判断：
1. 证据是否充分回答问题 (is_sufficient)
2. 证据覆盖度 (coverage_score)
3. 证据相关性和准确性 (specificity_score)
4. 是否存在矛盾信息 (has_contradiction)
5. 缺失的方面 (missing_aspects)
"""

from typing import List
from pydantic import BaseModel, Field
from .base import BaseAgent
from ..state import AgenticRAGState, RetrievalResult
from ..prompts.evidence_judge_prompts import (
    EVIDENCE_JUDGE_SYSTEM_MESSAGE,
    build_evidence_judgment_prompt
)
from ..utils.save_workflow_outputs import save_top10_chunks


class EvidenceJudgment(BaseModel):
    """
    Evidence Judge的评估结果
    
    判断标准：
    - is_sufficient: 证据是否足够回答问题（综合考虑数量、质量、覆盖度）
    - coverage_score: 证据对问题各方面的覆盖程度（0.0-1.0）
    - specificity_score: 证据的特定性和准确性（0.0-1.0）
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
        
        # Initialize retrieval tools for cross-encoder reranking
        from ..tools.retrieval_tools import RetrievalTools
        self.retrieval_tools = RetrievalTools(config)
    
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
            state: Contains question, question_type, retrieved_chunks, cpt_descriptions
            
        Returns:
            dict: Contains evidence_assessment
        """
        question = state["question"]
        question_type = state.get("question_type", "general")
        chunks = state.get("retrieved_chunks", [])
        retrieval_metadata = state.get("retrieval_metadata", {})
        cpt_descriptions = state.get("cpt_descriptions", {})  # Get CPT descriptions from state
        
        # 如果没有检索到内容 - 明确insufficient
        if not chunks:
            return {
                "evidence_assessment": {
                    "is_sufficient": False,
                    "coverage_score": 0.0,
                    "specificity_score": 0.0,
                    "has_contradiction": False,
                    "missing_aspects": ["No chunks retrieved - all aspects missing"],
                    "reasoning": "No relevant chunks were retrieved. Need to refine query or adjust retrieval strategy."
                }
            }
        
        # Apply cross-encoder reranking if enabled
        reranked_chunks = chunks  # Keep original for comparison
        if self.config.use_cross_encoder_rerank and len(chunks) > self.config.cross_encoder_top_k:
            print(f"\n🔄 Layer 3 Reranking: Cross-Encoder (Question-aware)")
            print(f"   Purpose: Refine {len(chunks)} chunks to top {self.config.cross_encoder_top_k} based on original question")
            print(f"   Before: {len(chunks)} chunks (from Layer 1-2 fusion)")
            
            # Call cross-encoder reranking tool
            reranked_chunks = self.retrieval_tools.cross_encoder_rerank(
                query=question,
                chunks=chunks,
                top_k=self.config.cross_encoder_top_k
            )
            
            print(f"   After: {len(reranked_chunks)} chunks (optimized for Evidence Judge)")
            
            # Update metadata
            retrieval_metadata["cross_encoder_reranked"] = True
            retrieval_metadata["cross_encoder_model"] = self.config.cross_encoder_model
            retrieval_metadata["chunks_before_layer3"] = len(chunks)
            retrieval_metadata["chunks_after_layer3"] = len(reranked_chunks)
            
            # 保存top 10 chunks作为LLM回答的依据
            mode = retrieval_metadata.get('mode', 'unknown')
            save_path = save_top10_chunks(
                top10_chunks=reranked_chunks,
                question=state.get('question', ''),
                output_dir=self.config.retrieval_output_dir,
                metadata={
                    'mode': mode,
                    'original_chunks_count': len(chunks),
                    'reranked_to_top': len(reranked_chunks),
                    'layer': 'layer3_cross_encoder'
                }
            )
            print(f"   💾 Top 10 chunks saved to: {save_path}")
        else:
            # Cross-encoder disabled or not enough chunks - use score-based top-K
            if not self.config.use_cross_encoder_rerank:
                print(f"\n⏭️  Layer 3 Reranking: Skipped (disabled in config)")
            else:
                print(f"\n⏭️  Layer 3 Reranking: Skipped (only {len(chunks)} chunks, threshold is {self.config.cross_encoder_top_k})")
            
            # Still limit to top-K based on existing scores (from Layer 1-2)
            if len(chunks) > self.config.cross_encoder_top_k:
                reranked_chunks = chunks[:self.config.cross_encoder_top_k]
                print(f"   📊 Using top {self.config.cross_encoder_top_k} chunks based on Layer 1-2 scores")
                retrieval_metadata["cross_encoder_reranked"] = False
                retrieval_metadata["truncated_to_top_k"] = True
            else:
                print(f"   📊 Using all {len(chunks)} chunks (no truncation needed)")
                retrieval_metadata["cross_encoder_reranked"] = False
            
            # 保存top 10 chunks作为LLM回答的依据 (score-based版本)
            mode = retrieval_metadata.get('mode', 'unknown')
            save_path = save_top10_chunks(
                top10_chunks=reranked_chunks,
                question=state.get('question', ''),
                output_dir=self.config.retrieval_output_dir,
                metadata={
                    'mode': mode,
                    'original_chunks_count': len(chunks),
                    'reranked_to_top': len(reranked_chunks),
                    'layer': 'layer1_layer2_score_based'
                }
            )
            print(f"   💾 Top {len(reranked_chunks)} chunks saved to: {save_path}")

        
        # Use reranked chunks for evaluation
        chunks_to_judge = reranked_chunks
        
        # 构建prompt用于LLM评估
        # 注意：只用 original question 和 retrieved chunks 评估
        # 不需要 sub-queries（它们只是检索手段，不是评估目标）
        prompt = self._build_judgment_prompt(
            question=question,
            question_type=question_type,
            chunks=chunks_to_judge,
            retrieval_metadata=retrieval_metadata,
            cpt_descriptions=cpt_descriptions  # Pass CPT descriptions to prompt builder
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
        
        # Return updated state with reranked chunks
        return {
            "evidence_assessment": {
                "is_sufficient": judgment.is_sufficient,
                "coverage_score": judgment.coverage_score,
                "specificity_score": judgment.specificity_score,
                "has_contradiction": judgment.has_contradiction,
                "missing_aspects": judgment.missing_aspects,
                "reasoning": judgment.reasoning
            },
            "retrieved_chunks": reranked_chunks,  # Update state with top-10 reranked chunks
            "retrieval_metadata": retrieval_metadata  # Update metadata with Layer 3 info
        }
    
    def _build_judgment_prompt(
        self,
        question: str,
        question_type: str,
        chunks: List[RetrievalResult],
        retrieval_metadata: dict,
        cpt_descriptions: dict = None
    ) -> str:
        """
        构建Evidence Judge的评估prompt
        
        评估逻辑：
        - 评估目标：original question（不是sub-queries）
        - 评估证据：retrieved chunks + CPT descriptions（已融合）
        - 评估标准：question_type 对应的 required aspects
        
        Args:
            question: Original user question（评估目标）
            question_type: Question type
            chunks: Retrieved chunks（已融合的15-20个chunks）
            retrieval_metadata: Retrieval metadata
            cpt_descriptions: CPT code -> description mapping (from retrieval)
        """
        # Format chunks
        chunks_text = self._format_chunks_for_evaluation(chunks)
        
        # Format CPT descriptions (if available)
        cpt_desc_text = ""
        if cpt_descriptions:
            cpt_desc_text = "\n\n### 📋 CPT Code Definitions (Retrieved)\n\n"
            for code, description in cpt_descriptions.items():
                cpt_desc_text += f"**CPT {code}**: {description}\n\n"
        
        # Extract metadata
        retrieval_mode = retrieval_metadata.get("mode", "unknown")
        strategies_used = retrieval_metadata.get("strategies_used", "N/A")
        
        # Use centralized prompt builder
        return build_evidence_judgment_prompt(
            question=question,
            question_type=question_type,
            chunks_text=chunks_text + cpt_desc_text,  # Append CPT descriptions to chunks
            retrieval_mode=retrieval_mode,
            strategies_used=str(strategies_used),
            total_chunks=len(chunks)
        )
    
    def _format_chunks_for_evaluation(self, chunks: List[RetrievalResult]) -> str:
        """
        分层展示 top 10 chunks + LLM总结截断部分，避免信息丢失
        
        策略（分层展示 + LLM总结）：
        - Chunk 1-5 (Detailed): 前800字符 + LLM 2-3句总结剩余部分（如果被截断）
        - Chunk 6-10 (Medium): 前400字符 + LLM 1-2句总结剩余部分（如果被截断）
        - 批量调用：1次LLM调用处理所有截断部分的总结
        
        Args:
            chunks: List of RetrievalResult objects (top 10 after Layer 3 reranking)
        """
        if not chunks:
            return "No chunks retrieved."
        
        # Step 1: 收集所有需要总结的截断部分
        chunks_to_summarize = []
        for i, chunk in enumerate(chunks[:10]):  # Only top 10 after Layer 3 reranking
            chunk_id, score, text, metadata = self._extract_chunk_data(chunk)
            
            # 根据tier决定展示长度（只有 top 10）
            if i < 5:
                preview_len = 800
                tier = "detailed"  # Chunk 1-5: 800 chars + 2-3 sentence summary
            else:
                preview_len = 400
                tier = "medium"  # Chunk 6-10: 400 chars + 1-2 sentence summary
            
            # 如果文本被截断，记录需要总结的部分
            if len(text) > preview_len:
                chunks_to_summarize.append({
                    "index": i,
                    "tier": tier,
                    "preview": text[:preview_len],
                    "remaining": text[preview_len:],  # 被截断的部分
                    "chunk_id": chunk_id,
                    "score": score,
                    "metadata": metadata
                })
            else:
                chunks_to_summarize.append({
                    "index": i,
                    "tier": tier,
                    "preview": text,
                    "remaining": None,  # 无需总结
                    "chunk_id": chunk_id,
                    "score": score,
                    "metadata": metadata
                })
        
        # Step 2: 批量总结所有截断部分（1次LLM调用）
        summaries = self._batch_summarize_truncated_parts(chunks_to_summarize)
        
        # Step 3: 格式化展示（将总结直接append到preview后面）
        formatted_chunks = []
        
        for item in chunks_to_summarize:
            i = item["index"]
            tier = item["tier"]
            chunk_id = item["chunk_id"]
            score = item["score"]
            metadata = item["metadata"]
            preview = item["preview"]
            summary = summaries.get(i, "")  # 获取该chunk的总结
            
            # 将总结直接拼接到preview后面，形成完整文本
            full_text = f"{preview} {summary}" if summary else preview
            
            cpt_info = f" [CPT: {metadata.get('cpt_code')}]" if metadata.get("cpt_code") else ""
            
            # 根据tier格式化（总结已append到full_text中）
            if tier == "detailed":
                formatted_chunks.append(
                    f"**[Detailed {i+1}]** (ID: {chunk_id}, Score: {score:.4f}){cpt_info}\n"
                    f"{full_text}"
                )
            else:
                formatted_chunks.append(
                    f"**[Medium {i+1}]** (Score: {score:.4f}){cpt_info}\n"
                    f"{full_text}"
                )
        
        # Step 4: 整体统计信息
        summary_info = f"\n\n**📊 Overall Summary**\n"
        summary_info += f"Total chunks analyzed: {len(chunks)}\n"
        summary_info += f"Score range: {chunks[0].score:.4f} (highest) to {chunks[len(chunks)-1].score:.4f} (lowest)"
        
        return "\n\n".join(formatted_chunks) + summary_info
    
    def _batch_summarize_truncated_parts(self, chunks_data: List[dict]) -> dict:
        """
        批量总结所有被截断的chunk部分（1次LLM调用）
        
        使用配置的主模型进行总结
        
        Args:
            chunks_data: List of chunk data with 'remaining' text to summarize
            
        Returns:
            dict: {chunk_index: summary_text}
        """
        # 收集需要总结的chunks
        to_summarize = [
            (item["index"], item["tier"], item["remaining"])
            for item in chunks_data
            if item["remaining"]  # 只总结被截断的部分
        ]
        
        if not to_summarize:
            return {}  # 无需总结
        
        # 构建批量总结prompt
        prompt = "Summarize the continuation of each chunk below. Be concise but capture key medical coding details.\n\n"
        
        for idx, tier, remaining_text in to_summarize:
            # 根据tier决定总结长度和要求
            if tier == "detailed":
                instruction = "Provide detailed summary (2-3 sentences covering key medical coding details)"
            else:
                instruction = "Provide medium summary (1-2 sentences with main points)"
            
            prompt += f"Chunk {idx} ({instruction}):\n{remaining_text[:1500]}\n\n"
        
        prompt += "\nIMPORTANT: Return summaries in format: Chunk X: [summary]\nEach summary should be a natural continuation of the previous text, without redundant introductions."
        
        try:
            # 使用同一个client和deployment（不需要单独的小模型）
            response = self.client.chat.completions.create(
                model=self.config.azure_deployment_name,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.3,
                max_tokens=1000
            )
            
            # 解析summaries
            summaries_text = response.choices[0].message.content
            summaries = self._parse_batch_summaries(summaries_text)
            
            return summaries
            
        except Exception as e:
            # 如果LLM调用失败，返回空（graceful degradation）
            print(f"Warning: Batch summarization failed: {e}")
            return {}
    
    def _parse_batch_summaries(self, summaries_text: str) -> dict:
        """
        解析批量总结的输出
        
        Expected format:
        Chunk 0: This section explains...
        Chunk 5: The remaining text discusses...
        """
        import re
        summaries = {}
        
        # 正则提取 "Chunk X: summary"
        pattern = r'Chunk\s+(\d+):\s*(.+?)(?=Chunk\s+\d+:|$)'
        matches = re.findall(pattern, summaries_text, re.DOTALL)
        
        for chunk_idx, summary in matches:
            summaries[int(chunk_idx)] = summary.strip()
        
        return summaries
    
    def _extract_chunk_data(self, chunk) -> tuple:
        """提取chunk数据（兼容dict和对象）"""
        if isinstance(chunk, dict):
            return (
                chunk.get("chunk_id", "unknown"),
                chunk.get("score", 0.0),
                chunk.get("text", ""),
                chunk.get("metadata", {})
            )
        else:
            return (chunk.chunk_id, chunk.score, chunk.text, chunk.metadata)

   
    