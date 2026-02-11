"""
测试Simple Agentic RAG Workflow
================================

验证整个agent pipeline从头到尾是否正常工作：
User Query → Orchestrator → Query Planner → Retrieval Router → Evidence Judge

注意：Workflow自动保存执行结果到memory/目录
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import AgenticRAGConfig
from src.workflow_simple import SimpleAgenticRAGWorkflow


def test_simple_workflow():
    """测试简化workflow的完整流程"""
    
    print("="*80)
    print("🧪 Testing Simple Agentic RAG Workflow")
    print("="*80)
    
    # 从环境变量加载配置
    config = AgenticRAGConfig.from_env()
    
    print(f"\n📋 Configuration:")
    print(f"   Retrieval Mode: {config.retrieval_mode}")
    print(f"   Top K: {config.top_k}")
    print(f"   Memory Dir: {config.memory_dir}")
    
    # 创建workflow（自动启用memory）
    workflow = SimpleAgenticRAGWorkflow(config, enable_memory=True)
    
    # 测试问题
    test_question = "What scenarios can't be reported with CPT code 44180?"
    test_cpt_code = 44180
    
    print(f"\n❓ Test Question: {test_question}")
    print(f"🏥 CPT Code: {test_cpt_code}")
    
    try:
        # 运行workflow（自动保存到memory）
        result = workflow.run(question=test_question)
        
        print("\n" + "="*80)
        print("📊 Final State Summary")
        print("="*80)
        
        # Show retry information
        retry_count = result.get('retry_count', 0)
        if retry_count > 0:
            print(f"\n🔄 Retry Information:")
            print(f"   Total Retries: {retry_count}")
            print(f"   Total Rounds: {retry_count + 1} (initial + {retry_count} retry)")
        
        print(f"\n1️⃣  Orchestrator:")
        print(f"   Question Type: {result.get('question_type')}")
        print(f"   Complexity: {result.get('question_complexity')}")
        print(f"   Strategy Hints: {result.get('retrieval_strategies')}")
        
        print(f"\n2️⃣  Query Planner:")
        query_candidates = result.get('query_candidates', [])
        print(f"   Query Candidates: {len(query_candidates)}")
        for i, qc in enumerate(query_candidates, 1):
            # qc is a QueryCandidate object
            query_text = qc.query if hasattr(qc, 'query') else str(qc)
            print(f"      {i}. {query_text}")
        
        print(f"\n3️⃣  Retrieval Router:")
        chunks = result.get('retrieved_chunks', [])
        print(f"   Retrieved Chunks: {len(chunks)}")
        metadata = result.get('retrieval_metadata', {})
        print(f"   Strategies Used: {metadata.get('strategies_used', 'N/A')}")
        
        print(f"\n4️⃣  Evidence Judge:")
        assessment = result.get('evidence_assessment', {})
        print(f"   Is Sufficient: {assessment.get('is_sufficient')}")
        print(f"   Coverage: {assessment.get('coverage_score', 0):.2f}")
        print(f"   Specificity: {assessment.get('specificity_score', 0):.2f}")
        
        if not assessment.get('is_sufficient'):
            missing = assessment.get('missing_aspects', [])
            if missing:
                print(f"   Missing Aspects ({len(missing)}):")
                for aspect in missing:
                    print(f"      • {aspect}")
        
        # Show Answer Generator output if available
        final_answer = result.get('final_answer')
        if final_answer:
            print(f"\n5️⃣  Answer Generator:")
            print(f"   Answer Preview: {final_answer.get('answer', 'N/A')[:150]}...")
            print(f"   Key Points: {len(final_answer.get('key_points', []))}")
            citation_map = final_answer.get('citation_map', {})
            print(f"   Citations: {len(citation_map)} chunks")
            print(f"   Confidence: {final_answer.get('confidence', 0):.2f}")
            if final_answer.get('limitations'):
                print(f"   Limitations: {len(final_answer.get('limitations', []))} noted")
        else:
            print(f"\n5️⃣  Answer Generator:")
            print(f"   Skipped (evidence insufficient)")
        
        print("\n" + "="*80)
        print("✅ All steps completed successfully!")
        print("="*80)
        
        # 验证关键字段
        checks = [
            ("Orchestrator set question_type", result.get('question_type') is not None),
            ("Orchestrator set complexity", result.get('question_complexity') is not None),
            ("Orchestrator provided strategy hints", len(result.get('retrieval_strategies', [])) > 0),
            ("Query Planner generated queries", len(query_candidates) > 0),
            ("Retrieval Router returned chunks", len(chunks) > 0),
            ("Evidence Judge provided assessment", assessment.get('is_sufficient') is not None),
            ("Coverage score in valid range", 0 <= assessment.get('coverage_score', -1) <= 1),
            ("Specificity score in valid range", 0 <= assessment.get('specificity_score', -1) <= 1),
        ]
        
        # Add answer generator check if evidence was sufficient
        if assessment.get('is_sufficient'):
            checks.append(("Answer Generator provided answer", final_answer is not None))
            # Check citation_map (dict) instead of citations (list)
            checks.append(("Answer has citations", len(final_answer.get('citation_map', {})) > 0 if final_answer else False))
        
        print("\n📋 Validation Checks:")
        all_passed = True
        for i, (desc, passed) in enumerate(checks, 1):
            status = "✅" if passed else "❌"
            print(f"   [{i}] {status} {desc}")
            all_passed = all_passed and passed
        
        if all_passed:
            print("\n🎉 All validation checks passed!")
            return True
        else:
            print("\n⚠️  Some validation checks failed!")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during workflow execution: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_modes():
    """测试不同的retrieval模式"""
    
    print("\n" + "="*80)
    print("🔄 Testing Different Retrieval Modes")
    print("="*80)
    
    modes = ["direct", "planning"]  # tool_calling需要更多LLM调用
    test_question = "What is CPT code 14301?"
    
    results = {}
    
    for mode in modes:
        print(f"\n{'='*80}")
        print(f"Testing mode: {mode}")
        print('='*80)
        
        try:
            config = AgenticRAGConfig.from_env()
            config.retrieval_mode = mode
            
            workflow = SimpleAgenticRAGWorkflow(config, enable_memory=True)
            result = workflow.run(question=test_question)
            
            metadata = result.get('retrieval_metadata', {})
            assessment = result.get('evidence_assessment', {})
            
            results[mode] = {
                'success': True,
                'chunks': len(result.get('retrieved_chunks', [])),
                'strategies': metadata.get('strategies_used', []),
                'coverage': assessment.get('coverage_score', 0),
                'specificity': assessment.get('specificity_score', 0)
            }
            
            print(f"✅ {mode} mode completed")
            print(f"   Chunks: {results[mode]['chunks']}")
            print(f"   Coverage: {results[mode]['coverage']:.2f}")
            
        except Exception as e:
            print(f"❌ {mode} mode failed: {e}")
            results[mode] = {'success': False, 'error': str(e)}
    
    # 对比结果
    print("\n" + "="*80)
    print("📊 Mode Comparison")
    print("="*80)
    
    for mode, data in results.items():
        if data['success']:
            print(f"\n{mode.upper()}:")
            print(f"  Chunks: {data['chunks']}")
            print(f"  Strategies: {data['strategies']}")
            print(f"  Coverage: {data['coverage']:.2f}")
            print(f"  Specificity: {data['specificity']:.2f}")
        else:
            print(f"\n{mode.upper()}: ❌ Failed - {data['error']}")
    
    return all(r['success'] for r in results.values())


if __name__ == "__main__":
    # Test 1: 基础workflow测试
    success = test_simple_workflow()
    
    # Test 2: 多模式对比测试（可选）
    # Uncomment to run mode comparison
    # success = success and test_multiple_modes()
    
    sys.exit(0 if success else 1)
