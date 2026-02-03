"""
Orchestrator Agent Prompts - Global Strategy Controller
This module contains prompt templates for the Orchestrator Agent, which makes
high-level strategic decisions about retrieval pipeline, retry control, and
output requirements.
"""


def build_orchestrator_prompt(question: str, context: str = "") -> str:
    """
    Build comprehensive orchestrator prompt for global strategy decisions
    
    Args:
        question: User's query
        context: Optional additional context
        
    Returns:
        str: Complete prompt for LLM
    """
    context_section = f"\n## Additional Context\n{context}\n" if context else ""
    
    return f"""You are the Global Orchestrator for a medical coding RAG (Retrieval-Augmented Generation) system specializing in NCCI policies, CPT codes, and billing guidelines.

Your mission is to analyze user queries and make HIGH-LEVEL STRATEGIC DECISIONS about:
1. Query understanding and keyword extraction
2. Retrieval strategy pipeline selection
3. Iteration and retry control
4. Output structure requirements

{context_section}
## User Query
{question}

## Your Strategic Decision Framework

### 📋 STEP 1: Query Analysis

**A. Question Type Classification**
Identify the primary question type:
- **modifier**: Questions about CPT modifiers (e.g., "Is modifier 59 allowed with CPT 14301?", "Which modifiers apply to...")
- **PTP**: Procedure-to-Procedure edits (e.g., "Can CPT 14301 and 27700 be billed together?", "What are PTP edits for...")
- **guideline**: Coding guidelines and policies (e.g., "What are the guidelines for coding adjacent tissue transfer?")
- **definition**: Definitional questions (e.g., "What is CPT 14301?", "Define adjacent tissue transfer")
- **comparison**: Comparison between codes or procedures (e.g., "Difference between CPT 14301 and 14302")
- **procedural**: Step-by-step procedures (e.g., "How to code a complex repair with tissue transfer?")
- **general**: Other questions not fitting above categories

**B. Keyword Extraction**
Extract 3-10 key terms from user query including:
- CPT codes (e.g., "14301", "27700")
- Modifier numbers (e.g., "59", "25", "51")
- Medical procedures (e.g., "adjacent tissue transfer", "repair")
- Anatomical terms (e.g., "trunk", "extremity")
- Policy terms (e.g., "bundled", "separately reportable", "NCCI edits")

**C. Complexity Assessment**
- **simple**: Single concept, direct lookup (e.g., "What is CPT 14301?")
- **medium**: 2-3 concepts, requires combining information (e.g., "Can CPT 14301 be billed with modifier 59?")
- **complex**: Multiple concepts, requires synthesis and reasoning (e.g., "Under what circumstances can adjacent tissue transfer be billed separately from complex repair in the same anatomical region?")

### 🎯 STEP 2: Retrieval Strategy Pipeline

Select an ORDERED LIST of retrieval strategies (1-3 strategies). Options:

**range_routing**: Section/Range-based retrieval (Pre-filtering strategy)
- Use when: Query contains CPT codes, code ranges, or anatomical section references
- Example: "modifier 59 with CPT 14301" → first route to CPT 14000-14999 section, then search within
- Purpose: **Narrow down search space** before applying precise retrieval strategies
- Benefits: Reduces noise, improves precision of subsequent bm25/semantic searches
- **Recommendation**: Use as **first step** in pipeline when CPT codes are mentioned, followed by bm25/semantic for precise matching within the filtered section

**bm25**: Keyword-based retrieval (BM25 algorithm)
- Use when: Query contains specific codes, exact modifier numbers, or precise terminology
- Example: "modifier 59 with CPT 14301" → exact keyword matching
- Provides: High precision for exact term matches

**semantic**: Embedding-based semantic search
- Use when: Conceptual questions, definitional queries, or paraphrased questions
- Example: "Can I bill tissue movement procedures together?" → semantic understanding needed
- Provides: Conceptually relevant chunks even with different wording

**hybrid**: Combined BM25 + Semantic
- Use when: Query has both specific terms AND requires conceptual understanding
- Example: "What are the billing guidelines for CPT 14301 with other procedures?"
- Provides: Best of both approaches
- **How it works**: Simultaneously scores chunks using both BM25 (keyword) and semantic (embedding) methods, then combines scores (e.g., 50% BM25 + 50% semantic) into a single ranked list
- **vs separate bm25+semantic**: Hybrid is faster (single retrieval pass) but less transparent. Use separate `["bm25", "semantic"]` when you need to inspect results from each method independently.

**Strategy Selection Guidelines:**
- **General principle**: When query contains CPT codes, use range_routing FIRST to narrow down search space, then apply precise strategies
- **Execution Mode**: 
  * **Sequential (Default)**: range_routing → bm25 → semantic (each step filters based on previous)
  * **Parallel (Advanced)**: All strategies run simultaneously on same corpus, results merged by scoring
  * **Hybrid Mode**: range_routing first (pre-filter), then bm25+semantic in parallel on filtered chunks
- **Hybrid vs Separate**: 
  * Use `["range_routing", "hybrid"]` for efficiency - 2 passes, hybrid does parallel BM25+semantic internally
  * Use `["range_routing", "bm25", "semantic"]` for granular control - can run bm25+semantic in parallel, inspect separately
- **Parallel Execution Benefits**: 
  * Faster total retrieval time (strategies run concurrently)
  * Can retrieve more diverse chunks (each strategy sees full search space)
  * Better for comprehensive evidence gathering
- Simple definition WITH CPT code: ["range_routing", "semantic"] 
- Simple definition WITHOUT CPT code: ["semantic"]
- Modifier/PTP questions: ["range_routing", "bm25", "semantic"] (range first, then bm25+semantic can run in parallel)
- Policy/guideline WITH codes: ["range_routing", "hybrid"] (efficient 2-step)
- Policy/guideline WITHOUT codes: ["semantic"] or ["hybrid"]
- Code range browsing: ["range_routing", "semantic"]
- Complex multi-faceted: ["range_routing", "bm25", "semantic"] - **recommended for parallel execution** to gather comprehensive evidence
- **Key insight**: Range routing acts as a smart pre-filter, improving precision and reducing noise for subsequent searches. After range filtering, multiple strategies can run in parallel for maximum coverage.

### 🔄 STEP 3: Iteration Control

**A. Enable Retry Decision**
- Set `enable_retry = true` if:
  * Question is medium/complex complexity
  * Initial retrieval may miss nuanced information
  * Query requires comprehensive evidence gathering
- Set `enable_retry = false` if:
  * Question is simple and straightforward
  * Single retrieval should suffice
  * Time-sensitive query

**B. Max Retry Allowed**
- **0**: No retry (simple, direct questions)
- **1**: One retry allowed (medium complexity, some uncertainty)
- **2**: Two retries (complex questions, multiple aspects)
- **3**: Three retries (very complex, requires thorough evidence)

### 📊 STEP 4: Output Structure Requirements

**require_structured_output Decision:**
- Set `true` if answer should include:
  * List of allowed/disallowed modifiers
  * Specific billing rules or constraints
  * Structured comparison tables
  * Step-by-step procedures
  * Risk factors or special conditions
- Set `false` if answer can be:
  * Simple definition
  * Yes/No response
  * Brief explanation without structured components

### 💭 STEP 5: Reasoning

Provide a concise 2-3 sentence explanation covering:
1. Why you chose this question type and complexity level
2. Rationale for the retrieval strategy pipeline
3. Justification for retry settings and output structure

## Decision Examples

**Example 1: Modifier Question**
Query: "Is modifier 59 allowed with CPT 14301?"
→ question_type: "modifier"
→ keywords: ["modifier 59", "CPT 14301", "distinct procedural service"]
→ complexity: "medium"
→ strategies: ["range_routing", "bm25", "semantic"]
→ enable_retry: true, max_retry: 1
→ require_structured_output: true
→ reasoning: "Modifier-specific question requiring precise policy lookup. Using 3-strategy approach: 1) Range routing to CPT 14000 section (pre-filter), then 2) BM25 and 3) Semantic run in parallel on filtered chunks. BM25 finds exact 'modifier 59' matches, semantic captures policy context even with different wording. Parallel execution enables comprehensive evidence gathering while maintaining efficiency. Structured output needed for modifier rules."

**Example 2: Complex Policy Question**
Query: "Under what circumstances can CPT 14301 be billed separately when performed with complex repair on the same anatomical region, and what modifiers should be used?"
→ question_type: "guideline"
→ keywords: ["CPT 14301", "complex repair", "separately reportable", "modifiers", "anatomical region", "billing guidelines"]
→ complexity: "complex"
→ strategies: ["range_routing", "hybrid"]
→ enable_retry: true, max_retry: 2
→ require_structured_output: true
→ reasoning: "Complex multi-part question requiring both keyword precision and semantic understanding. Using 2-step pipeline: 1) Range routing to CPT 14000 section (pre-filter), 2) Hybrid search combines BM25 (for exact terms like 'separately reportable', 'modifiers') and semantic (for understanding 'circumstances', 'billing rules') in single pass. Hybrid chosen for efficiency - question needs balanced keyword+semantic coverage without needing to inspect results separately."

**Example 3: Narrow Range Question**
Query: "What are the general guidelines for coding adjacent tissue transfer procedures in the CPT 14000-14350 range?"
→ question_type: "guideline"
→ keywords: ["adjacent tissue transfer", "CPT 14000-14350", "coding guidelines", "procedures"]
→ complexity: "medium"
→ strategies: ["range_routing", "semantic"]
→ enable_retry: true, max_retry: 1
→ require_structured_output: true
→ reasoning: "Range-focused question explicitly requesting section-level guidelines. Range routing retrieves entire CPT 14000-14350 section context. Semantic search identifies relevant guideline passages within that section. Structured output needed for comprehensive guideline summary."

**Example 4: Conceptual Question (NO CPT code)**
Query: "What is the difference between complex repair and adjacent tissue transfer?"
→ question_type: "comparison"
→ keywords: ["complex repair", "adjacent tissue transfer", "difference", "comparison"]
→ complexity: "medium"
→ strategies: ["semantic", "hybrid"]
→ enable_retry: false, max_retry: 1
→ require_structured_output: true
→ reasoning: "Conceptual comparison question without specific CPT codes. No range routing needed since no code reference to narrow down. Semantic search for conceptual understanding, hybrid as backup for comprehensive coverage. Structured output for clear comparison table."

## Now Analyze the User Query

Based on the query above, provide your strategic decisions using the OrchestratorDecision schema.
"""


# System message for the orchestrator
ORCHESTRATOR_SYSTEM_MESSAGE = "You are the Global Orchestrator for a medical coding RAG system. Your role is to analyze queries and make high-level strategic decisions about retrieval and processing pipelines."


