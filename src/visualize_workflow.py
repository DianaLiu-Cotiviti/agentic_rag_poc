"""
Visualize the Agentic RAG workflow graph
Generates a graphical representation of the LangGraph workflow
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_rag import AgenticRAGWorkflow, AgenticRAGConfig
from dotenv import load_dotenv


def visualize_workflow():
    """Generate and save workflow visualization"""
    
    load_dotenv()
    config = AgenticRAGConfig.from_env()
    workflow = AgenticRAGWorkflow(config)
    
    # Get the compiled graph
    graph = workflow.graph
    
    try:
        # Generate Mermaid diagram
        mermaid = graph.get_graph().draw_mermaid()
        
        print("📊 Agentic RAG Workflow Graph (Mermaid)")
        print("=" * 80)
        print(mermaid)
        print("=" * 80)
        
        # Save to file
        output_path = Path(__file__).parent.parent / "workflow_graph.md"
        with open(output_path, "w") as f:
            f.write("# Agentic RAG Workflow Graph\n\n")
            f.write("```mermaid\n")
            f.write(mermaid)
            f.write("\n```\n")
        
        print(f"\n✅ Workflow graph saved to: {output_path}")
        print("\nYou can visualize this in:")
        print("  - GitHub (renders Mermaid automatically)")
        print("  - VS Code (with Mermaid extension)")
        print("  - https://mermaid.live/")
        
    except Exception as e:
        print(f"❌ Error generating visualization: {e}")
        print("\nWorkflow structure:")
        print("  START → Orchestrator → Query Planner → Retrieval → Evidence Judge")
        print("                                                           ↓")
        print("                                                    [Sufficient?]")
        print("                                                     ↙          ↘")
        print("                                         Query Refiner      Structured Extraction → END")
        print("                                                ↓")
        print("                                            Retrieval")


if __name__ == "__main__":
    visualize_workflow()
