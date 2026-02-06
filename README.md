# Agentic RAG System 🤖

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-green.svg)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A production-ready multi-agent Retrieval-Augmented Generation (RAG) system built with LangGraph, specifically designed for NCCI medical coding policy document Q&A.

## ✨ Key Features

- 🧠 **Multi-Agent Collaboration**: 5 specialized agents working intelligently together
- 🔍 **Triple Retrieval Modes**: Direct (fast) / Planning (balanced) / Tool Calling (intelligent)
- 🎯 **Hybrid Retrieval**: Range Routing + BM25 + Semantic Vector + RRF Fusion
- 📊 **Evidence Quality Assessment**: Automated evidence sufficiency evaluation with retry mechanism
- 💾 **Complete Memory System**: Full execution history with timestamp-based storage
- 🔄 **Auto Index Management**: Intelligent index checking and building
- 📦 **Structured Output**: Answers + citations + confidence scores + execution logs

## 🏗️ System Architecture

### Workflow Overview

```
User Question
    ↓
🔧 Preprocessing: Auto-check and build indexes (Range + BM25 + ChromaDB)
    ↓
🧠 Orchestrator Agent (Intent Analysis, Strategy Hints)
    ↓
📋 Query Planner Agent (Generate 4 Query Candidates)
    ↓
🔍 Retrieval Router (3 Modes: Direct/Planning/Tool Calling)
    ├─ Direct Mode: Fixed pipeline (0 LLM calls, ~0.5s)
    ├─ Planning Mode: LLM-generated plan (1 LLM call, ~2s)
    └─ Tool Calling Mode: Agentic iteration (5-15 LLM calls, ~10s)
    ↓
⚖️  Evidence Judge Agent (Quality Assessment)
    ↓
💾 Memory System (Save complete execution history)
    ↓
Final Results + Retrieved Chunks
```

### Three Retrieval Modes

| Mode | LLM Calls | Speed | Cost | Intelligence | Use Case |
|------|-----------|-------|------|--------------|----------|
| **direct** | 0 | ~0.5s | $0 | ⚡ | Production (speed priority) |
| **planning** | 1 | ~2s | $0.01 | 🤖🤖 | Standard (balanced) |
| **tool_calling** | 5-15 | ~10s | $0.05+ | 🤖🤖🤖 | Research (quality priority) |

### Core Components

#### 1. Agents (`src/agents/`)

| Agent | File | Responsibility |
|-------|------|----------------|
| **Orchestrator** | `orchestrator.py` | Analyze question type, complexity, provide strategy hints |
| **Query Planner** | `query_planner.py` | Generate 4 query candidates (original, section-specific, synonym, constraint-focused) |
| **Retrieval Router** | `retrieval_router*.py` | Execute retrieval in 3 modes (direct/planning/tool_calling) |
| **Evidence Judge** | `evidence_judge.py` | Assess coverage, specificity, citations; identify missing aspects |

#### 2. Retrieval Tools (`src/tools/`)

| Tool | File | Function |
|------|------|----------|
| **Range Routing** | `retrieval_tools.py` | CPT code range filtering (SQLite-based) |
| **BM25 Search** | `bm25_store.py` | Lexical keyword search |
| **Semantic Search** | `chroma_store.py` | Vector similarity search (ChromaDB) |
| **Hybrid Search** | `retrieval_tools.py` | BM25 + Semantic RRF fusion |
| **Index Builder** | `build_indexes.py` | Auto-check and build all indexes |

#### 3. Configuration (`src/config.py`)

Centralized configuration with lazy client initialization:

```python
class AgenticRAGConfig:
    # Paths
    chunks_path: str = "rag/build/chunks.jsonl"
    range_index_path: str = "rag/build/cpt_range_index.db"
    bm25_index_path: str = "rag/build/bm25_index.pkl"
    chroma_db_path: str = "rag/build/chroma_db"
    
    # Retrieval mode
    retrieval_mode: str = "tool_calling"  # direct/planning/tool_calling
    
    # Lazy clients (shared across components)
    @property
    def client(self): ...  # Azure OpenAI client
    
    @property
    def embedding_client(self): ...  # Embedding client
    
    @property
    def chroma_client(self): ...  # ChromaDB client
```

#### 4. Memory System (`src/memory.py`)

Complete execution history with structured storage:

```python
WorkflowMemory.save_execution(
    question="What is CPT 14301?",
    final_state=state,
    workflow_type="simple",
    mode="tool_calling"
)
# Saves to: memory/workflow_simple_tool_calling_20260205_171201.json
#           memory/latest_simple_tool_calling.json
```

#### 5. Workflow Engine (`src/workflow_simple.py`)

Linear workflow (no retry) for testing and validation:

- Step 1: Orchestrator → Question analysis
- Step 2: Query Planner → Generate 4 candidates  
- Step 3: Retrieval Router → Execute retrieval
- Step 4: Evidence Judge → Quality assessment
- Auto-save: Memory + Retrieved chunks

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Azure OpenAI API access (with separate embedding endpoint)

### 1. Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd agentic_rag

# Activate virtual environment
source agentic_rag/bin/activate  # macOS/Linux
# or agentic_rag\Scripts\activate  # Windows

# Install dependencies (if needed)
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file with **two separate Azure OpenAI endpoints**:

```bash
# Chat/Completion Endpoint
AZURE_OPENAI_API_KEY=your-chat-api-key
AZURE_OPENAI_ENDPOINT=https://your-chat-endpoint.openai.azure.com/
AZURE_API_VERSION=2024-12-01-preview
AZURE_DEPLOYMENT_NAME=gpt-4o

# Embedding Endpoint (separate)
AZURE_OPENAI_API_KEY_EMBEDDING=your-embedding-api-key
AZURE_OPENAI_ENDPOINT_EMBEDDING=https://your-embedding-endpoint.openai.azure.com/
AZURE_API_VERSION_EMBEDDING=2024-02-15-preview
AZURE_DEPLOYMENT_NAME_EMBEDDING=text-embedding-3-large-2
```

### 3. Build Indexes (Auto-check on first run)

```bash
# Check and build missing indexes
python -m src.tools.build_indexes

# Force rebuild all indexes
python -m src.tools.build_indexes --force
```

Output:
```
📦 Checking and Building Indexes...
✓ Range Index already exists: rag/build/cpt_range_index.db
✓ BM25 Index already exists: rag/build/bm25_index.pkl
✓ ChromaDB Index already exists: rag/build/chroma_db
✅ All indexes ready!
```

### 4. Run Test Workflow

```bash
python test_workflow_simple.py
```

Sample output:
```
🧪 Testing Simple Agentic RAG Workflow
================================================================================

📋 Configuration:
   Retrieval Mode: tool_calling
   Top K: 15
   Memory Dir: memory

🔧 Preprocessing: Ensuring all indexes are built...
✅ All indexes ready!

🎯 Step 1: Orchestrator - Analyzing question...
Question Type: PTP
Complexity: medium
Strategy Hints: ['range_routing', 'bm25', 'semantic']

📋 Step 2: Query Planner - Generating query candidates...
Generated 4 query candidates:
  1. What is CPT code 14301 and when can it be billed with 27702?
  2. NCCI procedure-to-procedure edits for CPT 14301 and 27702
  3. Can adjacent tissue transfer CPT 14301 be billed together with tibial osteotomy CPT 27702?
  4. Billing restrictions and modifier indicators for CPT 14301 and 27702

🔍 Step 3: Retrieval Router - Executing retrieval...
Mode: tool_calling

  🔄 Tool Calling Iteration #1
     → range_routing(cpt_code=14301) → 50 chunks
     → range_routing(cpt_code=27702) → 15 chunks

  🔄 Tool Calling Iteration #2
     → bm25_search(...) → 20 chunks
     → semantic_search(...) → 20 chunks

  🔄 Tool Calling Iteration #3
     → rrf_fusion(result_ids=['bm25_0', 'semantic_0']) → 20 chunks

  ✅ LLM finished tool calling

📊 Tool Calling Execution Summary:
   Total iterations: 4
   Total tool calls: 5
   Final results: 20 chunks retrieved

⚖️  Step 4: Evidence Judge - Assessing evidence quality...
Is Sufficient: False
Coverage Score: 0.30
Specificity Score: 0.30

💾 Workflow result saved to: memory/workflow_simple_tool_calling_20260205_171201.json
💾 Retrieved chunks saved to: output/retrievals/retrieval_20260205_171201.json

✅ All validation checks passed!
```

## 📁 Project Structure

```
agentic_rag/
├── src/
│   ├── __init__.py                      # Package initialization
│   ├── config.py                        # ⚙️ Centralized configuration (paths, clients, settings)
│   ├── state.py                         # 📊 State definitions (AgenticRAGState TypedDict)
│   ├── memory.py                        # 💾 Memory system (WorkflowMemory class)
│   ├── workflow_simple.py               # 🔄 Simple workflow (linear, no retry)
│   ├── workflow.py                      # 🔄 Full workflow (with retry logic)
│   ├── agents_coordinator.py            # 🎭 Agent coordinator (facade pattern)
│   │
│   ├── agents/                          # 🤖 Agent implementations
│   │   ├── __init__.py
│   │   ├── base.py                      # Base agent class
│   │   ├── orchestrator.py              # Step 1: Question analysis
│   │   ├── query_planner.py             # Step 2: Query generation
│   │   ├── retrieval_router.py          # Step 3: Retrieval dispatcher
│   │   ├── retrieval_router_direct.py   # Direct mode (0 LLM calls)
│   │   ├── retrieval_router_planning.py # Planning mode (1 LLM call)
│   │   ├── retrieval_router_tool_calling.py # Tool calling mode (5-15 LLM calls)
│   │   └── evidence_judge.py            # Step 4: Evidence assessment
│   │
│   ├── tools/                           # 🔧 Retrieval and build tools
│   │   ├── __init__.py
│   │   ├── retrieval_tools.py           # Main retrieval tools (Range, BM25, Semantic, Hybrid)
│   │   ├── bm25_store.py                # BM25 index wrapper
│   │   ├── chroma_store.py              # ChromaDB wrapper
│   │   ├── build_indexes.py             # 📦 Unified index builder (auto-check)
│   │   ├── build_range_index.py         # Range index builder
│   │   ├── build_bm25.py                # BM25 index builder
│   │   └── build_embeddings_chroma.py   # ChromaDB embeddings builder
│   │
│   ├── prompts/                         # 📝 LLM prompt templates
│   │   ├── orchestrator.txt
│   │   ├── query_planner.txt
│   │   ├── evidence_judge.txt
│   │   └── ...
│   │
│   └── utils/                           # 🛠️ Utility functions
│       ├── keyword_parser.py
│       └── save_retrieval.py            # Save retrieved chunks
│
├── rag/
│   ├── build/                           # 🏗️ Built indexes
│   │   ├── chunks.jsonl                 # Processed chunks (~400 chunks)
│   │   ├── pages.jsonl                  # Page metadata
│   │   ├── table_of_contents.json       # TOC structure
│   │   ├── cpt_range_index.db           # Range routing index (SQLite, 2.4MB)
│   │   ├── bm25_index.pkl               # BM25 index (Pickle)
│   │   └── chroma_db/                   # ChromaDB vector store
│   │       ├── chroma.sqlite3           # Metadata (16MB, 481 embeddings)
│   │       └── <uuid-dirs>/             # Vector segments
│   │
│   └── data/                            # 📄 Source data
│       ├── raw/                         # Raw PDF files
│       └── processed/                   # Processed data
│
├── output/                              # 📤 Output directory
│   ├── queries/                         # Query execution logs
│   ├── evaluations/                     # Evaluation results
│   └── retrievals/                      # 💾 Retrieved chunks (JSON)
│       └── retrieval_20260205_171201.json
│
├── memory/                              # 💾 Workflow execution history
│   ├── workflow_simple_direct_*.json    # Direct mode executions
│   ├── workflow_simple_planning_*.json  # Planning mode executions
│   ├── workflow_simple_tool_calling_*.json  # Tool calling executions
│   ├── latest_simple_direct.json        # Latest direct mode
│   ├── latest_simple_planning.json      # Latest planning mode
│   └── latest_simple_tool_calling.json  # Latest tool calling
│
├── docs/                                # 📚 Documentation
│   ├── orchestrator_advantages_and_limitations.md
│   ├── retrieval_router_design.md
│   ├── retrieval_router_modes.md
│   ├── retrieval_strategy_execution_modes.md
│   ├── SIMPLE_WORKFLOW_ARCHITECTURE.md
│   └── tool_calling_patterns_comparison.md
│
├── testing/                             # 🧪 Test files
│   ├── test_retrieval_router_direct.py
│   ├── test_retrieval_router_planning.py
│   └── test_retrieval_router_tool_calling.py
│
├── test_workflow_simple.py              # 🧪 Main test script
├── test_build_indexes.py                # 🧪 Index building test
├── requirements.txt                     # 📦 Python dependencies
├── .env                                 # 🔐 Environment variables
└── README.md                            # 📖 This file
```

### Key Files

| File | Purpose |
|------|---------|
| `src/config.py` | Single source of truth for all paths and clients |
| `src/workflow_simple.py` | Main workflow orchestration |
| `src/memory.py` | Execution history management |
| `src/tools/build_indexes.py` | Auto-check and build all indexes |
| `test_workflow_simple.py` | Comprehensive workflow testing |

## 📚 Usage Examples

### Python API Usage

```python
from src.workflow_simple import SimpleAgenticRAGWorkflow
from src.config import AgenticRAGConfig

# Load configuration
config = AgenticRAGConfig.from_env()

# Create workflow (auto-checks indexes on init)
workflow = SimpleAgenticRAGWorkflow(config, enable_memory=True)

# Execute workflow
result = workflow.run(
    question="What is CPT code 14301 and when can it be billed with 27702?",
    cpt_code=14301
)

# Access results
print(f"Retrieved {len(result['retrieved_chunks'])} chunks")
print(f"Evidence sufficient: {result['evidence_assessment']['is_sufficient']}")
print(f"Coverage score: {result['evidence_assessment']['coverage_score']}")
```

### Change Retrieval Mode

Edit `src/config.py`:

```python
class AgenticRAGConfig(BaseModel):
    retrieval_mode: str = "direct"  # or "planning" or "tool_calling"
```

Or set environment variable:

```bash
export RETRIEVAL_MODE=planning
python test_workflow_simple.py
```

### Access Memory

```python
from src.memory import WorkflowMemory

memory = WorkflowMemory(memory_dir="memory")

# Load latest execution
latest = memory.load_latest(workflow_type="simple")

print(f"Question: {latest['metadata']['question']}")
print(f"Mode: {latest['retrieval']['retrieval_metadata']['mode']}")
print(f"Chunks: {latest['retrieval']['num_chunks']}")

# List history
history = memory.list_history(workflow_type="simple", limit=10)
for item in history:
    print(f"{item['timestamp']}: {item['question'][:50]}...")
```

### Analyze Retrieved Chunks

```python
import json

# Load retrieved chunks
with open("output/retrievals/retrieval_20260205_171201.json") as f:
    data = json.load(f)

print(f"Question: {data['question']}")
print(f"Mode: {data['metadata']['mode']}")
print(f"Total chunks: {data['num_chunks']}")

# Analyze chunks
for i, chunk in enumerate(data['chunks'][:5], 1):
    print(f"\nChunk {i} (score: {chunk['score']:.2f}):")
    print(f"  ID: {chunk['chunk_id']}")
    print(f"  Text: {chunk['text'][:100]}...")
```

## 🛠️ Advanced Configuration

### Retrieval Parameters

In `src/config.py`:

```python
# Retrieval settings
top_k: int = 15  # Final number of chunks to retrieve
retrieval_mode: str = "tool_calling"  # direct/planning/tool_calling

# Evidence judge thresholds
min_coverage_score: float = 0.7
min_specificity_score: float = 0.7
min_citation_count: int = 3

# Agent LLM settings
agent_temperature: float = 0
agent_max_tokens: int = 2000
```

### Build Index Settings

In `src/tools/build_embeddings_chroma.py`:

```python
BATCH_SIZE = 100  # Number of texts per batch
SLEEP_TIME = 0.5  # Sleep time between batches (seconds)
COLLECTION_NAME = "ncci_chunks"  # ChromaDB collection name
```

### Force Rebuild Indexes

```bash
# Rebuild all indexes
python -m src.tools.build_indexes --force

# Rebuild specific index
rm rag/build/bm25_index.pkl
python -m src.tools.build_indexes
```

## 📊 Performance Benchmarks

| Mode | LLM Calls | Avg Time | Avg Cost | Chunks Quality |
|------|-----------|----------|----------|----------------|
| Direct | 0 (retrieval) | 0.5s | $0 | ⭐⭐⭐ |
| Planning | 1 | 2-3s | $0.01 | ⭐⭐⭐⭐ |
| Tool Calling | 5-15 | 8-12s | $0.05-0.15 | ⭐⭐⭐⭐⭐ |

*Note: All modes use LLM for orchestrator, query planner, and evidence judge*

## 🔍 Detailed Execution Logs

### Memory File Structure

```json
{
  "metadata": {
    "timestamp": "2026-02-05T17:12:01",
    "question": "What is CPT code 14301...",
    "workflow_type": "simple",
    "success": true
  },
  "orchestrator": {
    "question_type": "PTP",
    "question_complexity": "medium",
    "retrieval_strategies": ["range_routing", "bm25", "semantic"]
  },
  "query_planner": {
    "num_candidates": 4,
    "query_candidates": [...]
  },
  "retrieval": {
    "num_chunks": 20,
    "retrieval_metadata": {
      "mode": "tool_calling",
      "execution_log": [
        {"iteration": 1, "tool_name": "range_routing", "chunks_returned": 50},
        {"iteration": 2, "tool_name": "bm25_search", "chunks_returned": 20},
        ...
      ],
      "saved_to": "output/retrievals/retrieval_20260205_171201.json"
    }
  },
  "evidence_judge": {
    "is_sufficient": false,
    "coverage_score": 0.30,
    "specificity_score": 0.30,
    "missing_aspects": [...]
  }
}
```

## 🧪 Testing

```bash
# Run main test
python test_workflow_simple.py

# Test index building
python test_build_indexes.py

# Test specific retrieval mode
# (Edit config.py retrieval_mode first)
python test_workflow_simple.py
```

## 📖 Documentation

- [Simple Workflow Architecture](docs/SIMPLE_WORKFLOW_ARCHITECTURE.md)
- [Retrieval Router Modes](docs/retrieval_router_modes.md)
- [Tool Calling Patterns](docs/tool_calling_patterns_comparison.md)
- [Orchestrator Design](docs/orchestrator_advantages_and_limitations.md)

## 📄 License

MIT License

## 📧 Contact

For questions or suggestions, please submit an Issue or contact the project maintainers.

---

**Built with ❤️ using LangGraph and Azure OpenAI**
