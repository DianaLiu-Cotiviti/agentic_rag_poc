# Agentic RAG System 🤖

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.30+-green.svg)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A multi-agent Retrieval-Augmented Generation (RAG) system built with LangGraph, specifically designed for NCCI medical coding policy document Q&A.

## ✨ Key Features

- 🧠 **Multi-Agent Collaboration**: 6 specialized agents working intelligently together
- 🔍 **Hybrid Retrieval Strategy**: BM25 + Semantic Vector + RRF Fusion
- 🎯 **Adaptive Strategy**: Automatically selects optimal retrieval approach based on question type
- 📊 **Evidence Quality Assessment**: Automatically judges evidence sufficiency and retries when necessary
- 📦 **Structured Output**: Complete answers + evidence citations + confidence scores
- 🔄 **Auditability**: Complete workflow logs with streaming execution support

## 🏗️ System Architecture

```
User Question
    ↓
🧠 Orchestrator Agent (Intent Analysis, Strategy Selection)
    ↓
🧭 Query Planner Agent (Generate Multiple Query Candidates)
    ↓
🔧 Retrieval Tools (Range + BM25 + Semantic + Hybrid)
    ↓
🧪 Evidence Judge Agent (Evidence Quality Assessment)
    ↓
🔁 Query Refiner Agent (Optional: Query Optimization Retry)
    ↓
📦 Structured Extraction Agent (Structured Output)
    ↓
Final Answer + Evidence
```

### Core Components

#### 1. Agents

| Agent | Function | Responsibility |
|-------|----------|----------------|
| **Orchestrator** | Strategy Orchestration | Parse user intent, determine question type and retrieval strategy |
| **Query Planner** | Query Planning | Generate multiple query candidates (original, expanded, synonyms, section-specific) |
| **Evidence Judge** | Evidence Assessment | Evaluate sufficiency and quality of retrieved evidence |
| **Query Refiner** | Query Optimization | Optimize queries for evidence gaps (retry mechanism) |
| **Structured Extraction** | Structured Extraction | Extract structured answers from evidence |

#### 2. Retrieval Tools

- **Range Routing**: CPT code range routing (SQLite-based indexing)
- **BM25 Search**: Lexical keyword retrieval
- **Semantic Search**: Semantic vector retrieval (ChromaDB)
- **Hybrid Search**: BM25 + Semantic hybrid retrieval (RRF fusion)
- **Multi-Query Search**: Multi-query candidate fusion retrieval

#### 3. Workflow Engine

State machine workflow built with **LangGraph**, supporting:
- ✅ Conditional branching (evidence sufficiency evaluation)
- ✅ Retry loops (max 2 attempts)
- ✅ State tracking (complete logging)
- ✅ Streaming execution (optional)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Azure OpenAI API access

### 1. Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd agentic_rag

# Create virtual environment (recommended)
python -m venv agentic_rag
source agentic_rag/bin/activate  # macOS/Linux
# or agentic_rag\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file:

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_API_VERSION=2024-02-15-preview
AZURE_DEPLOYMENT_NAME=gpt-4o
AZURE_DEPLOYMENT_NAME_EMBEDDING=text-embedding-3-large
```

### 3. Verify Installation

```bash
python test_setup.py
```

### 4. Usage

#### 🔹 Interactive Mode

```bash
python src/agentic_rag_cli.py --mode interactive
```

#### 🔹 Single Query Mode

```bash
python src/agentic_rag_cli.py --mode single \
    --question "What modifiers are allowed for CPT 31622?" \
    --cpt-code 31622
```

#### 🔹 Streaming Execution (View Intermediate Steps)

```bash
python src/agentic_rag_cli.py --mode single \
    --question "Can CPT 14301 be billed with modifier 59?" \
    --cpt-code 14301 \
    --stream
```

#### 🔹 Batch Processing Mode

Prepare input file `questions.json`:

```json
[
  {
    "question": "What modifiers are allowed for CPT 31622?",
    "cpt_code": "31622"
  },
  {
    "question": "Can CPT 27700 and 27701 be billed together?",
    "cpt_code": "27700"
  }
]
```

Run batch processing:

```bash
python src/agentic_rag_cli.py --mode batch \
    --input examples/sample_questions.json \
    --output results.json
```

## 📁 Project Structure

```
agentic_rag/
├── src/
│   ├── __init__.py               # Package entry point
│   ├── config.py                 # Configuration management
│   ├── state.py                  # State definitions
│   ├── agents.py                 # Agent nodes
│   ├── workflow.py               # LangGraph workflow
│   ├── agentic_rag_cli.py        # CLI main program
│   ├── example_agentic_rag.py    # Usage examples
│   ├── visualize_workflow.py     # Workflow visualization
│   ├── evaluation.py             # Evaluation tools
│   ├── experiment_tracker.py     # Experiment tracking
│   │
│   ├── tools/                    # 🔧 Retrieval tools module
│   │   ├── __init__.py
│   │   ├── retrieval_tools.py    # Agentic RAG retrieval tools
│   │   ├── bm25_store.py         # BM25 indexing
│   │   ├── chroma_store.py       # ChromaDB vector store
│   │   └── ...
│   │
│   └── prompts/                  # Prompt templates
│       ├── orchestrator.txt
│       ├── query_planner.txt
│       └── ...
│
├── examples/
│   └── sample_questions.json     # Sample questions
│
├── build/                        # Build artifacts
│   ├── chunks.jsonl
│   ├── pages.jsonl
│   ├── table_of_contents.json
│   └── chroma_db/
│
├── data/                         # Data files
├── output/                       # Output results
│
├── test_setup.py                 # System test script
├── test_workflow.py              # Workflow tests
├── quickstart.sh                 # Quick start script
├── requirements.txt              # Python dependencies
├── .env.template                 # Environment variable template
│
└── docs/                         # Documentation
    ├── AGENTIC_RAG_README.md     # Detailed README
    ├── USAGE_GUIDE.md            # Usage guide
    ├── AGENT_ARCHITECTURE.md     # Architecture documentation
    ├── PROJECT_SUMMARY.md        # Project summary
    └── QUICK_REFERENCE.md        # Quick reference
```

## 📚 Usage Examples

### Python API Usage

```python
from src.workflow import create_agentic_rag_graph
from src.state import AgenticRAGState

# Create workflow
graph = create_agentic_rag_graph()

# Prepare input
initial_state = AgenticRAGState(
    user_question="What modifiers are allowed for CPT 31622?",
    cpt_code="31622"
)

# Execute workflow
final_state = graph.invoke(initial_state)

# Get results
print(final_state["final_answer"])
print(final_state["structured_output"])
```

### Streaming Execution

```python
# Streaming execution, view each step
for event in graph.stream(initial_state):
    print(f"Step: {event}")
```

## 🛠️ Advanced Configuration

Configurable in [config.py](src/config.py):

```python
# Retrieval parameters
BM25_TOP_K = 10
SEMANTIC_TOP_K = 10
HYBRID_TOP_K = 15
MULTI_QUERY_TOP_K = 20

# Retry configuration
MAX_RETRIES = 2
RETRY_THRESHOLD = 0.6  # Evidence sufficiency threshold

# LLM parameters
TEMPERATURE = 0.1
MAX_TOKENS = 4096
```

## 📄 License

MIT License

## 📧 Contact

For questions or suggestions, please submit an Issue or contact the project maintainers.

---

**Built with ❤️ using LangGraph and Azure OpenAI**
