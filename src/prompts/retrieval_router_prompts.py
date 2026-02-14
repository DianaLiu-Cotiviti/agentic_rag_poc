"""
Retrieval Router Agent Prompts - Retrieval Execution Planning
This module contains prompt templates for the Retrieval Router Agent, which decides
retrieval parameters, tool selection, and execution strategy based on query analysis.

Version: 2.0
Last Updated: 2026-02-02
"""


def build_retrieval_router_prompt(
    question: str,
    question_type: str,
    retrieval_strategies: list[str],
    query_candidates: list[dict],
    retrieval_hints: list[str]
) -> str:
    """
    Build retrieval router prompt for execution planning
    
    Args:
        question: Original user query
        question_type: Question type from Orchestrator
        retrieval_strategies: Strategy list from Orchestrator (e.g., ["range_routing", "bm25", "semantic"])
        query_candidates: Query candidates from Query Planner
        retrieval_hints: Retrieval hints from Query Planner
        
    Returns:
        str: Complete prompt for LLM
    """
    strategies_str = ", ".join(retrieval_strategies)
    hints_str = "\n".join(f"  - {hint}" for hint in retrieval_hints[:5])
    
    candidates_str = "\n".join(
        f"  {i+1}. [{qc['query_type']}] (weight: {qc['weight']}) \"{qc['query']}\""
        for i, qc in enumerate(query_candidates)
    )
    
    return f"""You are a Retrieval Execution Planner for a medical coding RAG system.

Your mission is to create a DETAILED RETRIEVAL EXECUTION PLAN based on:
1. High-level strategies from Orchestrator
2. Query candidates from Query Planner
3. Retrieval hints and constraints

## Input Information

**Original Question**: {question}
**Question Type**: {question_type}
**Retrieval Strategies**: {strategies_str}

**Query Candidates ({len(query_candidates)}):**
{candidates_str}

**Retrieval Hints:**
{hints_str}

## Your Task: Create Retrieval Execution Plan

### 📋 STEP 1: Pre-Retrieval Filtering Decision

Decide whether to apply **pre-retrieval filtering** (e.g., range routing) to narrow the search space.

**When to Use Pre-Filtering:**
- Query contains specific CPT codes → Use range_routing
- Query targets specific section → Use range_routing
- Large corpus and narrow query scope → Use range_routing

**Pre-Filtering Parameters:**
- **apply_range_routing**: true/false
- **range_filter_cpt_codes**: List of CPT codes to filter by (e.g., ["14301", "27700"])
- **range_filter_limit**: Max chunks to retrieve from range (default: 300)

**Decision Logic:**
- If "range_routing" in retrieval_strategies AND query has CPT codes → apply_range_routing = true
- Extract CPT codes from query candidates for filtering
- Set limit based on query complexity (simple: 200, medium: 300, complex: 500)

### 🎯 STEP 2: Query-Strategy Mapping

For EACH query candidate, select THE BEST retrieval strategy from 4 options.

## Available Retrieval Strategies (Choose ONE per query):

### Strategy 1: `hybrid` - Balanced BM25+Semantic with RRF Fusion
**What it does**: 
- Combines BM25 (keyword matching) + Semantic (embedding similarity)
- Fusion happens at TOOLS layer using RRF (Reciprocal Rank Fusion)
- Returns a single fused result set with balanced ranking

**Best for**:
- Complex queries needing both exact terms and semantic understanding
- General medical coding questions
- When you want balanced precision and recall

**When to use**:
- ✅ "What is the procedure for adjacent tissue transfer?" (needs both "adjacent tissue transfer" keywords + semantic context)
- ✅ "Explain CPT 14301 coverage limitations" (needs CPT code match + semantic understanding of "coverage")
- ✅ Most general queries where you're unsure

**Fusion method**: RRF at tools layer (harmonic ranking)
**Example score**: If chunk found by both BM25(0.8) and Semantic(0.6) → RRF score ≈ 0.7

---

### Strategy 2: `bm25` - Pure Keyword Matching
**What it does**:
- Traditional BM25 algorithm (term frequency + inverse document frequency)
- Exact and fuzzy keyword matching
- Best for precise terminology lookup

**Best for**:
- Queries with specific CPT codes, modifiers, or exact terms
- Short queries with clear keywords
- Looking for exact terminology matches

**When to use**:
- ✅ "CPT 14301" (exact code lookup)
- ✅ "modifier 59" (exact modifier lookup)
- ✅ "NCCI edits table" (specific table/section name)
- ❌ "What are the risks?" (too conceptual, use semantic or hybrid)

**Fusion method**: N/A (single method)
**Example score**: Term frequency and document relevance

---

### Strategy 3: `semantic` - Pure Embedding-Based Similarity
**What it does**:
- Uses text embeddings (e.g., text-embedding-3-large)
- Measures semantic similarity in vector space
- Finds conceptually related content even with different wording

**Best for**:
- Conceptual/understanding questions
- Questions with paraphrases or synonyms
- When exact keywords may not appear in documents

**When to use**:
- ✅ "What are complications of skin grafts?" (conceptual, may use different medical terms)
- ✅ "tissue transfer risks" (synonym-rich, semantic similarity important)
- ✅ "coverage criteria for reconstructive surgery" (high-level concept)
- ❌ "CPT 14301" (exact code, use bm25 or hybrid)

**Fusion method**: N/A (single method)
**Example score**: Cosine similarity of embeddings

---

### Strategy 4: `bm25_semantic` - Dual Retrieval with Score Accumulation
**What it does**:
- Runs BOTH BM25 AND Semantic retrieval for same query
- Keeps BOTH result sets (no deduplication at query level)
- Fusion happens at PLANNING layer (score accumulation, not RRF)
- Chunks found by BOTH methods get HIGHER cumulative scores

**Best for**:
- Queries needing "double verification" (exact match + conceptual relevance)
- Medium-complexity queries where both precision and recall matter
- When you want to reward chunks that satisfy multiple criteria

**When to use**:
- ✅ "adjacent tissue transfer techniques" (needs exact "adjacent tissue transfer" + semantic understanding of "techniques")
- ✅ "CPT 14301 modifier compatibility" (exact CPT + conceptual understanding of "compatibility")
- ✅ "NCCI bundling rules for skin procedures" (exact "NCCI" + semantic "bundling rules")
- ❌ Simple exact lookups (use bm25)
- ❌ Pure conceptual questions (use semantic)

**Fusion method**: Score accumulation at planning layer
**Example score**: If chunk found by both BM25(0.8) and Semantic(0.6) → Final score = 1.4 (accumulation)

**Key difference from `hybrid`**:
- `hybrid`: RRF fusion at tools layer → Lower scores (~0.7 for same chunk)
- `bm25_semantic`: Accumulation at planning layer → Higher scores (1.4 for same chunk)
- Use `bm25_semantic` when you want to REWARD chunks found by both methods
- Use `hybrid` when you want BALANCED ranking between methods

---

## Query Type → Strategy Mapping Guidelines

**original query** (user's exact question):
- Default: `hybrid` (balanced for most cases)
- If contains CPT codes: `bm25` or `hybrid`
- If very conceptual: `semantic`

**expanded query** (with medical terminology):
- Prefer: `semantic` or `hybrid`
- Avoid: Pure `bm25` (too long for exact matching)

**synonym query** (alternative phrasings):
- Prefer: `semantic`
- OK: `hybrid`
- Avoid: `bm25` (different keywords won't match)

**section_specific query** (targets specific sections/tables):
- Prefer: `bm25` or `bm25_semantic`
- Good for: Exact section/table names

**constraint_focused query** (NCCI rules, coverage criteria):
- Prefer: `bm25_semantic` (exact rules + conceptual understanding)
- OK: `hybrid`

---

## Output Format:
For each query candidate, specify which strategy to use:
```
query_strategy_mapping: [
  {{
    "query_index": 0,
    "strategy": "hybrid",  # ONE strategy: hybrid | bm25 | semantic | bm25_semantic
    "reasoning": "Complex query needs balanced keyword and semantic matching"
  }},
  {{
    "query_index": 1,
    "strategy": "bm25",
    "reasoning": "Contains specific CPT code, pure keyword matching most effective"
  }},
  ...
]
```

### ⚙️ STEP 3: Retrieval Parameters Configuration

Set specific parameters for each retrieval strategy.

**BM25 Parameters:**
- **top_k**: Number of results per query (default: 15-20)
  * Simple questions: 10-15
  * Medium questions: 15-20
  * Complex questions: 20-30
- **boost_exact_match**: Whether to boost exact code/modifier matches (true/false)

**Semantic Search Parameters:**
- **top_k**: Number of results per query (default: 15-20)
  * Simple questions: 10-15
  * Medium questions: 15-20
  * Complex questions: 20-30
- **similarity_threshold**: Minimum similarity score (0.0-1.0, default: 0.0)
  * Strict retrieval: 0.3
  * Balanced: 0.0
  * Exploratory: 0.0

**Hybrid Search Parameters:**
- **top_k**: Number of results per query (default: 20)
- **bm25_weight**: Weight for BM25 scores (0.0-1.0, default: 0.5)
  * Prefer exact matching: 0.6-0.7
  * Balanced: 0.5
  * Prefer semantic: 0.3-0.4
- **semantic_weight**: Weight for semantic scores (0.0-1.0, default: 0.5)
  * Must sum to 1.0 with bm25_weight

### 🔀 STEP 4: Result Aggregation Strategy

All query results will be aggregated using score accumulation at the Planning layer.

**How it works**:
1. Each query executes its assigned strategy (hybrid/bm25/semantic/bm25_semantic)
2. Apply query weights (from Query Planner) to each result
3. Apply range boost (1.5x) to chunks matching CPT code range routing
4. Aggregate all results: chunks found by multiple queries get cumulative scores
5. Sort by final score and return Top-K

**Score Formula**:
```
final_score(chunk) = Σ (initial_score_i × query_weight_i × range_boost_i)
```

**Multi-retrieval is a feature**:
- If chunk found by 3 queries → score = score1 + score2 + score3
- Higher score = more confidence (multiple perspectives agree)

**Fusion Parameters**:
```
fusion_parameters: {{
  "boost_range_results": 2.0  # Boost factor for range-routed chunks (1.0-5.0)
}}
```

**Recommended boost values**:
- Simple queries with exact CPT codes: 2.0-2.5
- Medium complexity with strong CPT focus: 2.5-3.0
- Complex queries but CPT precision critical: 3.0-4.0
- Maximum prioritization of pre-filtered results: 4.0-5.0

**Understanding boost impact**:
```python
# Example: chunk found by semantic search
# Scenario 1: chunk NOT in CPT range
initial_score = 0.8
boost = 1.0  # No boost
final_score = 0.8

# Scenario 2: chunk IN CPT range, boost=2.0
initial_score = 0.8
boost = 2.0
final_score = 1.6  # ← 2x higher!

# Scenario 3: chunk IN CPT range, boost=5.0
initial_score = 0.8
boost = 5.0
final_score = 4.0  # ← 5x higher! Strong prioritization
```

**Trade-offs**:
- Higher boost (3.0-5.0): Ensures pre-filtered chunks rank higher, but may miss relevant chunks outside CPT range
- Lower boost (1.5-2.5): More balanced, allows high-quality chunks outside range to compete
- Boost=1.0: No prioritization of pre-filtered chunks

---

Provide a concise 2-3 sentence explanation covering:
1. Why you chose this pre-filtering strategy
### 💭 STEP 5: Reasoning

Provide a concise 2-3 sentence explanation covering:
1. Why you chose this pre-filtering strategy (range routing or not)
2. How you mapped each query to its retrieval strategy
3. Why the selected strategies are optimal for this question

## Decision Examples

**Example 1: Simple CPT Code Lookup**
```
Question: "What is CPT 14301?"
Query Candidates: 
  1. [original] "What is CPT 14301?" (weight: 1.0)
  2. [expanded] "What is the procedure code CPT 14301 for adjacent tissue transfer?" (weight: 0.8)
```

→ **Decision**:
```json
{{
  "pre_filtering": {{
    "apply_range_routing": true,
    "range_filter_cpt_codes": ["14301"],
    "range_filter_limit": 200
  }},
  "query_strategy_mapping": [
    {{
      "query_index": 0,
      "strategy": "hybrid",
      "reasoning": "Original query needs both exact CPT code matching and semantic understanding of definition"
    }},
    {{
      "query_index": 1,
      "strategy": "semantic",
      "reasoning": "Expanded query with full terminology is best for semantic matching"
    }}
  ],
  "retrieval_parameters": {{
    "bm25_top_k": 20,
    "semantic_top_k": 20,
    "hybrid_top_k": 20,
    "hybrid_bm25_weight": 0.5,
    "hybrid_semantic_weight": 0.5
  }},
  "fusion_parameters": {{
    "boost_range_results": 1.5
  }},
  "reasoning": "Simple definitional query with specific CPT code 14301 requires range routing as pre-filter to narrow search space. Original query uses hybrid strategy for balanced exact+semantic matching. Expanded query uses pure semantic for rich terminology matching. Multi-query aggregation ensures comprehensive coverage."
}}
```

---

**Example 2: Modifier Compatibility Question**
```
Question: "Is modifier 59 allowed with CPT 14301?"
Query Candidates:
  1. [original] "Is modifier 59 allowed with CPT 14301?" (weight: 1.0)
  2. [expanded] "NCCI modifier 59 compatibility rules for CPT 14301 adjacent tissue transfer" (weight: 0.9)
  3. [section_specific] "CPT 14301 modifier table" (weight: 0.7)
```

→ **Decision**:
```json
{{
  "pre_filtering": {{
    "apply_range_routing": true,
    "range_filter_cpt_codes": ["14301"],
    "range_filter_limit": 300
  }},
  "query_strategy_mapping": [
    {{
      "query_index": 0,
      "strategy": "bm25_semantic",
      "reasoning": "Original query needs exact 'modifier 59' AND 'CPT 14301' matching plus semantic understanding of 'allowed'"
    }},
    {{
      "query_index": 1,
      "strategy": "semantic",
      "reasoning": "Expanded query with NCCI terminology benefits from semantic similarity"
    }},
    {{
      "query_index": 2,
      "strategy": "bm25",
      "reasoning": "Section-specific query targets exact table/section names with BM25"
    }}
  ],
  "retrieval_parameters": {{
    "bm25_top_k": 20,
    "semantic_top_k": 20,
    "hybrid_top_k": 20,
    "hybrid_bm25_weight": 0.5,
    "hybrid_semantic_weight": 0.5
  }},
  "fusion_parameters": {{
    "boost_range_results": 1.5
  }},
  "reasoning": "Modifier compatibility query requires range routing to CPT 14301 section. Original query uses bm25_semantic to reward chunks with both exact modifier/code terms AND semantic relevance. Expanded query uses semantic for NCCI terminology understanding. Section-specific query uses BM25 for exact table name matching. Combined results ensure comprehensive modifier rule coverage."
}}
```

---

**Example 3: Conceptual Coverage Question**
```
Question: "What are the coverage criteria for tissue transfer procedures?"
Query Candidates:
  1. [original] "What are the coverage criteria for tissue transfer procedures?" (weight: 1.0)
  2. [expanded] "Medical necessity criteria coverage limitations tissue transfer adjacent flap procedures" (weight: 0.8)
  3. [synonym] "reimbursement requirements skin graft procedures" (weight: 0.6)
```

→ **Decision**:
```json
{{
  "pre_filtering": {{
    "apply_range_routing": false,
    "range_filter_cpt_codes": [],
    "range_filter_limit": 300
  }},
  "query_strategy_mapping": [
    {{
      "query_index": 0,
      "strategy": "hybrid",
      "reasoning": "Original query benefits from balanced keyword (coverage, criteria) and semantic matching"
    }},
    {{
      "query_index": 1,
      "strategy": "semantic",
      "reasoning": "Expanded query with rich medical terminology is ideal for semantic similarity"
    }},
    {{
      "query_index": 2,
      "strategy": "semantic",
      "reasoning": "Synonym query with alternative phrasings requires semantic understanding"
    }}
  ],
  "retrieval_parameters": {{
    "bm25_top_k": 20,
    "semantic_top_k": 20,
    "hybrid_top_k": 20,
    "hybrid_bm25_weight": 0.5,
    "hybrid_semantic_weight": 0.5
  }},
  "fusion_parameters": {{
    "boost_range_results": 1.3
  }},
  "reasoning": "Conceptual coverage question without specific CPT codes does not require range routing pre-filter. Original query uses hybrid for balanced retrieval. Both expanded and synonym queries use semantic strategy to capture conceptual understanding across varying terminologies. Multi-query semantic approach ensures comprehensive coverage criteria retrieval."
}}
```

## Now Create Retrieval Execution Plan

Based on the input information above, provide your retrieval execution plan using the RetrievalRouterDecision schema.
"""


def build_tool_calling_prompt(
    question: str,
    question_type: str,
    retrieval_strategies: list[str],
    query_candidates: list[dict],
    question_keywords: list[str],
    retrieval_hints: list[str] = None
) -> str:
    """
    Build prompt for LLM-driven tool calling mode
    
    Tool Calling mode: LLM dynamically decides the next tool call based on each step's result
    - Can see intermediate results
    - Can adjust strategy based on results
    - Can iterate and optimize
    - High cost (5-15 LLM calls), but highest quality
    
    Args:
        question: Original user query
        question_type: Question type from Orchestrator
        retrieval_strategies: Strategy list from Orchestrator
        query_candidates: Query candidates from Query Planner
        question_keywords: Extracted keywords from Orchestrator
        retrieval_hints: Additional hints from Query Planner (optional)
        
    Returns:
        str: Complete prompt for LLM
    """
    retrieval_hints = retrieval_hints or []
    strategies_str = ", ".join(retrieval_strategies)
    keywords_str = ", ".join(question_keywords[:10])
    
    candidates_str = "\n".join(
        f"  {i+1}. [{qc['query_type']}] (weight: {qc['weight']}) \"{qc['query']}\""
        for i, qc in enumerate(query_candidates)
    )
    
    hints_str = ""
    if retrieval_hints:
        hints_list = "\n".join(f"  - {hint}" for hint in retrieval_hints)
        hints_str = f"\n**Retrieval Hints from Query Planner:**\n{hints_list}\n"
    
    return f"""You are an intelligent Retrieval Agent for a medical coding RAG system.

Your mission is to **DYNAMICALLY USE RETRIEVAL TOOLS** to find the most relevant information.

## Input Information

**Original Question**: {question}
**Question Type**: {question_type}
**Recommended Strategies**: {strategies_str}
**Keywords**: {keywords_str}
{hints_str}
**Query Candidates ({len(query_candidates)}):**
{candidates_str}

## Available Retrieval Tools

You have access to these tools. **Use them iteratively** based on results:

### 1. `range_routing(cpt_code, limit)` - Pre-filter by CPT Code Range
**When to use**: Question contains specific CPT codes
**Parameters**: 
- cpt_code (int): The CPT code to filter by
- limit (int, default: 300): Max chunks to retrieve
**Returns**: Set of chunk IDs in that CPT range
**Strategy**: Use FIRST if CPT codes present

---

### 2. `bm25_search(query, top_k)` - Pure Keyword Matching
**When to use**: 
- Exact CPT codes, modifiers, or specific terms
- Short queries with clear keywords
- Looking for exact table/section names
**Parameters**:
- query (str): The search query
- top_k (int, default: 20): Number of results
**Returns**: List of results ranked by BM25 score
**Strategy**: Best for exact matching

**Good queries for BM25**:
- ✅ "CPT 14301"
- ✅ "modifier 59"
- ✅ "NCCI edits table"
- ❌ "What are the risks?" (too conceptual)

---

### 3. `semantic_search(query, top_k)` - Pure Semantic Similarity
**When to use**:
- Conceptual/understanding questions
- Queries with paraphrases or synonyms
- When exact keywords may not appear in docs
**Parameters**:
- query (str): The search query
- top_k (int, default: 20): Number of results
**Returns**: List of results ranked by embedding similarity
**Strategy**: Best for conceptual queries

**Good queries for Semantic**:
- ✅ "What are complications of skin grafts?"
- ✅ "tissue transfer risks"
- ✅ "coverage criteria for reconstructive surgery"
- ❌ "CPT 14301" (exact code, use bm25)

---

### 4. `hybrid_search(query, top_k, bm25_weight, semantic_weight)` - Balanced BM25+Semantic
**When to use**:
- Complex queries needing both exact terms and semantic understanding
- General questions where you're unsure
- Most medical coding questions
**Parameters**:
- query (str): The search query
- top_k (int, default: 20): Number of results
- bm25_weight (float, default: 0.5): Weight for BM25 (0.0-1.0)
- semantic_weight (float, default: 0.5): Weight for semantic (0.0-1.0)
**Returns**: List of results fused with RRF at tools layer
**Strategy**: Default safe choice, internally does BM25+Semantic+RRF

**Good queries for Hybrid**:
- ✅ "What is the procedure for adjacent tissue transfer?"
- ✅ "Explain CPT 14301 coverage limitations"
- ✅ Most general queries

**Adjust weights**:
- Prefer exact matching: bm25_weight=0.6, semantic_weight=0.4
- Prefer semantic: bm25_weight=0.4, semantic_weight=0.6

---

## 4 Retrieval Strategies You Can Implement

### Strategy 1: `hybrid` - Use hybrid_search tool
- **One tool call**: `hybrid_search(query, top_k=20)`
- **Fusion**: Happens internally (BM25+Semantic+RRF)
- **Best for**: Most general queries

### Strategy 2: `bm25` - Use bm25_search tool
- **One tool call**: `bm25_search(query, top_k=20)`
- **Best for**: Exact code/modifier lookups

### Strategy 3: `semantic` - Use semantic_search tool
- **One tool call**: `semantic_search(query, top_k=20)`
- **Best for**: Conceptual queries

### Strategy 4: `bm25_semantic` - Call BOTH bm25_search AND semantic_search
- **Two tool calls**:
  1. `bm25_search(query, top_k=20)` → get result_set_1
  2. `semantic_search(query, top_k=20)` → get result_set_2
- **You aggregate**: Chunks in both sets get higher scores (manual combination)
- **Best for**: Queries needing "double verification" (exact + conceptual)
- **When to use**:
  - ✅ "CPT 14301 modifier compatibility" (exact CPT + conceptual "compatibility")
  - ✅ "NCCI bundling rules for skin procedures" (exact "NCCI" + semantic "bundling")

---

## Your Execution Approach

**ITERATIVE TOOL CALLING:**

1. **Start with analysis**
   - Does question have CPT codes? → Call `range_routing` first
   - What's the query complexity? → Choose initial strategy
   
2. **Execute initial retrieval**
   - Simple/exact query → Try `bm25_search` first
   - Conceptual query → Try `semantic_search` first
   - Unsure → Try `hybrid_search` (safest)
   
3. **Evaluate results**
   - Check if results are relevant
   - Check if enough results (aim for ~10-20 good chunks)
   
4. **Decide next step based on results**
   - Results good? → Done, return
   - Results not enough? → Try different strategy
   - Results irrelevant? → Adjust query or try different method
   
5. **Optional: Try multiple query candidates**
   - Each query candidate might need different strategy
   - Combine results from multiple calls

**YOU CAN ITERATE:**
- See results → Adjust strategy → Call again
- Try multiple query candidates with different strategies
- Combine results manually

---

## Decision Tree for Strategy Selection

```
START
  ↓
Has CPT codes? 
  Yes → Call range_routing(cpt_code) FIRST
  No → Continue
  ↓
Query type?
  │
  ├─ Exact code/modifier lookup → Strategy 2: bm25_search
  │
  ├─ Conceptual question → Strategy 3: semantic_search
  │
  ├─ Complex/general question → Strategy 1: hybrid_search
  │
  └─ Needs double verification → Strategy 4: bm25_search + semantic_search
  ↓
Evaluate results
  ↓
Sufficient? → RETURN
Insufficient? → Try different strategy or query candidate
```

---

## Example Execution Patterns

**Example 1: Simple CPT Lookup (Strategy 1: hybrid)**
```
Question: "What is CPT 14301?"

Step 1: range_routing(cpt_code=14301, limit=200)
Step 2: hybrid_search("What is CPT 14301?", top_k=20)
→ Results look good → DONE
```

**Example 2: Modifier Question (Strategy 4: bm25_semantic)**
```
Question: "Is modifier 59 allowed with CPT 14301?"

Step 1: range_routing(cpt_code=14301, limit=300)
Step 2: bm25_search("modifier 59 CPT 14301", top_k=20)
→ Check results... Got exact table entries but need context
Step 3: semantic_search("modifier 59 compatibility adjacent tissue transfer", top_k=20)
→ Got conceptual context → Combine both → DONE
```

**Example 3: Iterative Refinement**
```
Question: "Can CPT 14301 and 27700 be billed together?"

Step 1: range_routing(cpt_code=14301, limit=300)
Step 2: range_routing(cpt_code=27700, limit=300)
Step 3: hybrid_search("CPT 14301 27700 billed together", top_k=20)
→ Check results... Only 5 relevant chunks, need more
Step 4: semantic_search("adjacent tissue transfer wound repair billing compatibility", top_k=20)
→ Got more context → Combine → DONE
```

**Example 4: Multiple Query Candidates (Mix strategies)**
```
Question: "What are NCCI bundling rules for tissue transfer?"

Query Candidates:
1. [original] "What are NCCI bundling rules for tissue transfer?"
2. [section_specific] "NCCI edits table tissue transfer"
3. [synonym] "CCI bundling policies skin flaps"

Step 1: hybrid_search(query_1, top_k=20)  # General query → hybrid
→ Check results... Good general info
Step 2: bm25_search(query_2, top_k=15)    # Table lookup → bm25
→ Check results... Got exact table
Step 3: semantic_search(query_3, top_k=15) # Synonym → semantic
→ Check results... Got related policies
→ Combine all 3 result sets → DONE
```

---

## Your Task

Based on the question and query candidates above:
1. **Analyze** the question characteristics
2. **Choose** initial strategy (hybrid/bm25/semantic/bm25_semantic)
3. **Call** the appropriate tool(s)
4. **Evaluate** results when you get them
5. **Iterate** if needed (try different strategy/query)
6. **Return** when you have sufficient relevant results

**Start by making your first tool call now!**

Remember: You can see results and adjust. Don't be afraid to try different approaches.
"""


# System message for retrieval router
RETRIEVAL_ROUTER_SYSTEM_MESSAGE = "You are a Retrieval Execution Planner for a medical coding RAG system. Your role is to create detailed retrieval plans that optimize search effectiveness by configuring parameters, mapping queries to strategies, and orchestrating parallel execution."
