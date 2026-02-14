
# --- Standard imports ---
import os
import sys
import streamlit as st
import json
import hashlib
# Ensure project root is in sys.path for src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.config import AgenticRAGConfig
from src.workflow_simple import SimpleAgenticRAGWorkflow

st.set_page_config(page_title="Agentic RAG Workflow Viewer", layout="wide")

st.title("Agentic RAG Workflow Viewer")

# --- Workflow Graph Visualization ---
import graphviz
st.markdown("#### Agentic RAG Workflow Graph")
mode_labels = ["Direct", "LLM Planning", "LLM Tool Call"]
mode_map = {"Direct": "direct", "LLM Planning": "planning", "LLM Tool Call": "tool_calling"}
selected_label = st.radio("Select retrieval mode:", mode_labels, index=0)
mode = mode_map[selected_label]

def get_graphviz(mode):
    if mode == "direct":
        dot = graphviz.Digraph()
        dot.attr(rankdir="LR")
        dot.node("Q", "User Question", style="filled", fillcolor="#e3e3ff")
        dot.node("O", "Orchestrator\n(LLM)", style="filled", fillcolor="#f9f")
        dot.node("P", "Query Planner\n(LLM)", style="filled", fillcolor="#bbf")
        dot.node("RR", "Retrieval Router\n(Fixed)", style="filled", fillcolor="#bfb")
        dot.node("RG", "Range Routing\n(CPT)", style="filled", fillcolor="#bfb")
        dot.node("HS", "Hybrid Search\n(BM25+Semantic)", style="filled", fillcolor="#bfb")
        dot.node("FUS", "Multi-query RRF Fusion", style="filled", fillcolor="#bfb")
        dot.node("E", "Evidence Judge\n(LLM)", style="filled", fillcolor="#ffb")
        dot.node("A", "Answer Generator\n(LLM)", style="filled", fillcolor="#fbf")
        dot.edges([("Q","O"), ("O","P"), ("P","RR"), ("RR","RG"), ("RG","HS"), ("HS","FUS"), ("FUS","E"), ("E","A")])
        return dot
    elif mode == "planning":
        dot = graphviz.Digraph()
        dot.attr(rankdir="LR")
        dot.node("Q", "User Question", style="filled", fillcolor="#e3e3ff")
        dot.node("O", "Orchestrator\n(LLM)", style="filled", fillcolor="#f9f")
        dot.node("P", "Query Planner\n(LLM)", style="filled", fillcolor="#bbf")
        dot.node("RP", "Retrieval Planner\n(LLM)", style="filled", fillcolor="#f9f")
        dot.node("R1", "Hybrid/Semantic/BM25 Search", style="filled", fillcolor="#bfb")
        dot.node("FUS", "Weighted Fusion", style="filled", fillcolor="#bfb")
        dot.node("E", "Evidence Judge\n(LLM)", style="filled", fillcolor="#ffb")
        dot.node("AR", "Answer Generator\n(LLM)", style="filled", fillcolor="#fbf")
        dot.node("QR", "Query Refiner\n(LLM, Retry)", style="filled", fillcolor="#bbf")
        dot.edges([("Q","O"), ("O","P"), ("P","RP"), ("RP","R1"), ("R1","FUS"), ("FUS","E")])
        dot.edge("E","AR", label="sufficient")
        dot.edge("E","QR", label="insufficient")
        dot.edge("QR","RP")
        return dot
    elif mode == "tool_calling":
        dot = graphviz.Digraph()
        dot.attr(rankdir="LR")
        dot.node("Q", "User Question", style="filled", fillcolor="#e3e3ff")
        dot.node("O", "Orchestrator\n(LLM)", style="filled", fillcolor="#f9f")
        dot.node("P", "Query Planner\n(LLM)", style="filled", fillcolor="#bbf")
        dot.node("TC", "Tool Calling Loop\n(LLM Agent)", style="filled", fillcolor="#f9f")
        dot.node("T1", "range_routing", style="filled", fillcolor="#bfb")
        dot.node("T2", "bm25_search", style="filled", fillcolor="#bfb")
        dot.node("T3", "semantic_search", style="filled", fillcolor="#bfb")
        dot.node("T4", "hybrid_search", style="filled", fillcolor="#bfb")
        dot.node("FUS", "Multi-tool Fusion", style="filled", fillcolor="#bfb")
        dot.node("E", "Evidence Judge\n(LLM)", style="filled", fillcolor="#ffb")
        dot.node("AR", "Answer Generator\n(LLM)", style="filled", fillcolor="#fbf")
        dot.node("QR", "Query Refiner\n(LLM, Retry)", style="filled", fillcolor="#bbf")
        dot.edges([("Q","O"), ("O","P"), ("P","TC")])
        dot.edges([("TC","T1"), ("TC","T2"), ("TC","T3"), ("TC","T4")])
        dot.edges([("T1","FUS"), ("T2","FUS"), ("T3","FUS"), ("T4","FUS")])
        dot.edge("FUS","E")
        dot.edge("E","AR", label="sufficient")
        dot.edge("E","QR", label="insufficient")
        dot.edge("QR","TC")
        return dot
    else:
        return graphviz.Digraph()

st.graphviz_chart(get_graphviz(mode))
st.markdown("---")



# --- Session and cache logic ---
if 'last_mode' not in st.session_state:
    st.session_state['last_mode'] = None
if 'last_question' not in st.session_state:
    st.session_state['last_question'] = None
if 'result_cache' not in st.session_state:
    st.session_state['result_cache'] = None

# Compute a unique cache key for (mode, question)
def get_cache_key(mode, question):
    """
    Generate cache key based on mode and first 20 chars of question
    This allows:
    - Different modes to have separate cache (direct/planning/tool_calling)
    - Similar questions (same first 20 chars) to share cache within same mode
    - Different questions to have separate cache
    - Readable filenames with mode prefix + question prefix
    """
    # Get first 20 chars, strip and clean for filename
    question_prefix = question.strip()[:20]
    # Remove special characters for filename safety
    safe_prefix = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in question_prefix)
    safe_prefix = safe_prefix.strip().replace(' ', '_')
    
    # Generate short hash based on mode + first 20 chars
    key = f"{mode}::{question.strip()[:20]}"
    short_hash = hashlib.md5(key.encode('utf-8')).hexdigest()[:8]
    
    # Include mode in filename for clarity
    return f"{mode}_{safe_prefix}_{short_hash}"

def get_cache_path(cache_key):
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{cache_key}.json")


# Create tabs for different views
tab1, tab2 = st.tabs(["Run Workflow", "Compare Modes"])

# Tab 1: Original workflow execution
with tab1:
    question = st.text_input("Enter your question:", value="What scenarios can't be reported with CPT code 44180?", key="run_question")
    run_clicked = st.button("Run Workflow", key="run_button")
    
    cache_key = get_cache_key(mode, question)
    cache_path = get_cache_path(cache_key)

    # If mode/question changed, clear session cache
    if (st.session_state['last_mode'] != mode) or (st.session_state['last_question'] != question):
        st.session_state['result_cache'] = None
        st.session_state['last_mode'] = mode
        st.session_state['last_question'] = question

    result = None
    log_text = None


    if run_clicked:
        loaded_from_cache = False
        cache_load_error = False
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                result = cache_data.get('result')
                log_text = cache_data.get('log_text')
                st.session_state['result_cache'] = (result, log_text)
                loaded_from_cache = True
            except Exception as e:
                cache_load_error = True
                st.warning(f"Cache file is corrupted or invalid, will rerun. Error: {e}")
                try:
                    os.remove(cache_path)
                except Exception:
                    pass
        if not loaded_from_cache:
            import io
            import logging
            
            # Create a StringIO buffer for logging
            log_buffer = io.StringIO()
            
            # Create a custom logging handler to capture logs
            log_handler = logging.StreamHandler(log_buffer)
            log_handler.setLevel(logging.INFO)
            log_handler.setFormatter(logging.Formatter('%(message)s'))
            
            # Configure logging to capture all agenticrag logs
            # 1. Get parent logger and set propagation
            agenticrag_logger = logging.getLogger("agenticrag")
            agenticrag_logger.setLevel(logging.INFO)
            agenticrag_logger.propagate = True
            
            # 2. Add handler to parent logger to capture all child logger messages
            agenticrag_logger.addHandler(log_handler)
            
            # 3. Also ensure specific loggers propagate to parent
            specific_loggers = [
                "agenticrag.workflow_simple",
                "agenticrag.retrieval_tools",
                "agenticrag.evidence_judge",
                "agenticrag.query_refiner",
                "agenticrag.retrieval_router_planning",
                "agenticrag.retrieval_router_tool_calling"
            ]
            for logger_name in specific_loggers:
                specific_logger = logging.getLogger(logger_name)
                specific_logger.propagate = True
            
            try:
                config = AgenticRAGConfig.from_env()
                config.retrieval_mode = mode
                workflow = SimpleAgenticRAGWorkflow(config, enable_memory=True)
                with st.spinner(f"Running workflow in '{mode}' mode..."):
                    result = workflow.run(question=question)
                log_text = log_buffer.getvalue()
                st.session_state['result_cache'] = (result, log_text)
            finally:
                # Remove the handler after workflow completes
                agenticrag_logger.removeHandler(log_handler)
                log_handler.close()

        # Always save the latest result to cache (even if just ran)
        if result is not None and log_text is not None and not loaded_from_cache:
            def make_json_safe(obj):
                if isinstance(obj, list):
                    return [make_json_safe(x) for x in obj]
                elif hasattr(obj, 'dict'):
                    return obj.dict()
                elif isinstance(obj, dict):
                    return {k: make_json_safe(v) for k, v in obj.items()}
                else:
                    return obj
            safe_result = make_json_safe(result)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({'result': safe_result, 'log_text': log_text}, f, ensure_ascii=False, indent=2)

    # If not run this time, but have cache in session, use it
    if not run_clicked and st.session_state['result_cache']:
        result, log_text = st.session_state['result_cache']

    if result is not None:
        # Show retrieval mode
        mode = result.get('retrieval_mode', 'unknown')
        st.info(f"🔧 **Retrieval Mode:** `{mode}`")
        
        # Show captured logs (step-by-step agent process)
        with st.expander("📝 Agent Step-by-Step Log", expanded=True):
            st.code(log_text or '', language="text")

        st.subheader("Workflow Steps & LLM/Tool Responses")
        st.markdown("---")

        # 1. Orchestrator
        with st.expander("1️⃣ Orchestrator", expanded=True):
            st.write(f"**Question Type:** {result.get('question_type')}")
            st.write(f"**Complexity:** {result.get('question_complexity')}")
            st.write(f"**Strategy Hints:** {result.get('retrieval_strategies')}")
            # Show raw LLM response if available
            if 'orchestrator_llm_response' in result:
                st.code(result['orchestrator_llm_response'], language='json')

        # 2. Query Planner
        with st.expander("2️⃣ Query Planner", expanded=True):
            query_candidates = result.get('query_candidates', [])
            st.write(f"**Query Candidates:** {len(query_candidates)}")
            for i, qc in enumerate(query_candidates, 1):
                query_text = qc.query if hasattr(qc, 'query') else str(qc)
                st.write(f"{i}. {query_text}")
            if 'query_planner_llm_response' in result:
                st.code(result['query_planner_llm_response'], language='json')

        # 3. Retrieval Router
        with st.expander("3️⃣ Retrieval Router", expanded=True):
            chunks = result.get('retrieved_chunks', [])
            st.write(f"**Retrieved Chunks:** {len(chunks)}")
            metadata = result.get('retrieval_metadata', {})
            st.write(f"**Strategies Used:** {metadata.get('strategies_used', 'N/A')}")
            if 'retrieval_router_tool_response' in result:
                st.code(result['retrieval_router_tool_response'], language='json')
            if chunks:
                with st.expander("Show Retrieved Chunks", expanded=False):
                    for i, chunk in enumerate(chunks, 1):
                        st.write(f"Chunk {i}:")
                        st.code(str(chunk))

        # Main (final) Evidence Judge
        with st.expander("4️⃣ Evidence Judge", expanded=True):
            assessment = result.get('evidence_assessment', {})
            st.write(f"**Is Sufficient:** {assessment.get('is_sufficient')}")
            st.write(f"**Coverage:** {assessment.get('coverage_score', 0):.2f}")
            st.write(f"**Specificity:** {assessment.get('specificity_score', 0):.2f}")
            if not assessment.get('is_sufficient'):
                missing = assessment.get('missing_aspects', [])
                if missing:
                    st.write(f"**Missing Aspects ({len(missing)}):**")
                    for aspect in missing:
                        st.write(f"- {aspect}")
            if 'evidence_judge_llm_response' in result:
                st.code(result['evidence_judge_llm_response'], language='json')

        # 5. Answer Generator
        with st.expander("5️⃣ Answer Generator", expanded=True):
            final_answer = result.get('final_answer')
            if final_answer:
                answer = final_answer.get('answer', 'N/A')
                citation_map = final_answer.get('citation_map', {})
                key_points = final_answer.get('key_points', [])
                
                st.markdown(f"**Answer:**\n\n{answer}")
                
                # Display Key Points with content
                st.markdown(f"**Key Points:** ({len(key_points)} total)")
                if key_points:
                    for i, kp in enumerate(key_points, 1):
                        st.markdown(f"{i}. {kp}")
                else:
                    st.write("_(No key points provided)_")
                
                st.write(f"**Confidence:** {final_answer.get('confidence', 0):.2f}")
                # Show Limitations content if present
                limitations = final_answer.get('limitations', [])
                if limitations:
                    st.markdown("**Limitations:**")
                    for lim in limitations:
                        st.write(f"- {lim}")
                if citation_map:
                    st.markdown("---")
                    st.markdown("**Citations:**")
                    for num, chunk_info in citation_map.items():
                        if isinstance(chunk_info, dict):
                            chunk_id = chunk_info.get('chunk_id', 'N/A')
                            chunk_text = chunk_info.get('chunk_text', '')
                        else:
                            chunk_id = str(chunk_info)
                            chunk_text = '(No chunk text found. Only chunk_id available.)'
                        with st.expander(f"[{num}] {chunk_id}"):
                            st.code(chunk_text)
                if 'answer_generator_llm_response' in final_answer:
                    st.code(final_answer['answer_generator_llm_response'], language='json')
            else:
                st.write("Skipped (evidence insufficient)")

        st.markdown("---")
        st.success("All steps completed!")

# Tab 2: Compare Modes
with tab2:
    st.subheader("Mode Comparison")
    st.write("Compare results from all three retrieval modes for the same question.")
    
    compare_question = st.text_input("Question to compare:", value="What scenarios can't be reported with CPT code 44180?", key="compare_question")
    compare_clicked = st.button("Compare All Modes", key="compare_button")
    
    if compare_clicked:
        import io
        import logging
        
        modes_to_compare = ["direct", "planning", "tool_calling"]
        mode_results = {}
        
        # Check which modes have cache
        missing_modes = []
        for m in modes_to_compare:
            cache_key = get_cache_key(m, compare_question)
            cache_path = get_cache_path(cache_key)
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    mode_results[m] = cache_data.get('result')
                except Exception as e:
                    st.warning(f"Failed to load cache for {m} mode: {e}")
                    missing_modes.append(m)
            else:
                missing_modes.append(m)
        
        # Run workflows for missing modes
        if missing_modes:
            st.info(f"Running workflows for missing modes: {', '.join(missing_modes)}")
            
            for m in missing_modes:
                with st.spinner(f"Running {m} mode..."):
                    log_buffer = io.StringIO()
                    log_handler = logging.StreamHandler(log_buffer)
                    log_handler.setLevel(logging.INFO)
                    log_handler.setFormatter(logging.Formatter('%(message)s'))
                    
                    agenticrag_logger = logging.getLogger("agenticrag")
                    agenticrag_logger.setLevel(logging.INFO)
                    agenticrag_logger.propagate = True
                    agenticrag_logger.addHandler(log_handler)
                    
                    specific_loggers = [
                        "agenticrag.workflow_simple",
                        "agenticrag.retrieval_tools",
                        "agenticrag.evidence_judge",
                        "agenticrag.query_refiner",
                        "agenticrag.retrieval_router_planning",
                        "agenticrag.retrieval_router_tool_calling"
                    ]
                    for logger_name in specific_loggers:
                        specific_logger = logging.getLogger(logger_name)
                        specific_logger.propagate = True
                    
                    try:
                        config = AgenticRAGConfig.from_env()
                        config.retrieval_mode = m
                        workflow = SimpleAgenticRAGWorkflow(config, enable_memory=True)
                        result = workflow.run(question=compare_question)
                        log_text = log_buffer.getvalue()
                        mode_results[m] = result
                        
                        # Save to cache
                        def make_json_safe(obj):
                            if isinstance(obj, list):
                                return [make_json_safe(x) for x in obj]
                            elif hasattr(obj, 'dict'):
                                return obj.dict()
                            elif isinstance(obj, dict):
                                return {k: make_json_safe(v) for k, v in obj.items()}
                            else:
                                return obj
                        safe_result = make_json_safe(result)
                        cache_key = get_cache_key(m, compare_question)
                        cache_path = get_cache_path(cache_key)
                        with open(cache_path, 'w', encoding='utf-8') as f:
                            json.dump({'result': safe_result, 'log_text': log_text}, f, ensure_ascii=False, indent=2)
                    finally:
                        agenticrag_logger.removeHandler(log_handler)
                        log_handler.close()
            
            st.success("All modes completed!")
        
        # Helper function to highlight differences between texts
        def highlight_differences(texts, labels):
            """
            Compare multiple texts and highlight unique parts using word-level diff
            
            Strategy:
            1. Use word-level comparison instead of sentence-level (more precise)
            2. Only highlight words/phrases that are truly unique or significantly different
            3. Use lighter highlighting for better readability
            4. Filter out minor differences (like punctuation, articles)
            
            Returns HTML-highlighted texts where:
            - Yellow highlight = unique content not found in other modes
            """
            import difflib
            import re
            
            if len(texts) != 3 or not all(texts):
                return texts
            
            def tokenize(text):
                """Split text into words while preserving punctuation context"""
                # Split on whitespace but keep the structure
                return text.split()
            
            def get_semantic_words(text):
                """Extract semantically meaningful words (filter out common words)"""
                # Remove common words that don't carry much meaning
                stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                            'can', 'may', 'will', 'would', 'should', 'could', 'this', 'that',
                            'these', 'those', 'it', 'its', 'as', 'per', 'via'}
                words = re.findall(r'\b\w+\b', text.lower())
                return set(w for w in words if w not in stopwords and len(w) > 2)
            
            highlighted_texts = []
            
            for idx, text in enumerate(texts):
                if not text.strip():
                    highlighted_texts.append(text)
                    continue
                
                # Get other two texts for comparison
                other_texts = [texts[i] for i in range(3) if i != idx]
                combined_other = ' '.join(other_texts)
                
                # Extract semantic keywords from this text
                current_keywords = get_semantic_words(text)
                other_keywords = get_semantic_words(combined_other)
                
                # Find truly unique keywords (appear in current but not in others)
                unique_keywords = current_keywords - other_keywords
                
                # Only highlight if there are meaningful unique keywords
                if not unique_keywords:
                    highlighted_texts.append(text)
                    continue
                
                # Build highlighted text by highlighting unique keywords
                highlighted = text
                for keyword in unique_keywords:
                    # Use word boundary to avoid partial matches
                    # Case-insensitive replacement
                    pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                    highlighted = pattern.sub(
                        lambda m: f'<mark style="background-color: #fff9c4; padding: 1px 3px; border-radius: 2px; font-weight: 500;">{m.group()}</mark>',
                        highlighted
                    )
                
                highlighted_texts.append(highlighted)
            
            return highlighted_texts
        
        # Display comparison
        if len(mode_results) == 3:
            st.markdown("---")
            st.subheader("📊 Comparison Results")
            
            # Collect all answers and key points for comparison
            all_answers = []
            all_key_points_list = []
            all_key_points_text = []
            for m in modes_to_compare:
                result = mode_results[m]
                final_answer = result.get('final_answer', {})
                answer_text = final_answer.get('answer', '') if final_answer else ''
                key_points = final_answer.get('key_points', []) if final_answer else []
                all_answers.append(answer_text)
                all_key_points_list.append(key_points)
                # Convert key points to text for comparison
                key_points_text = '\n'.join(key_points) if key_points else ''
                all_key_points_text.append(key_points_text)
            
            # Highlight differences in answers and key points
            highlighted_answers = highlight_differences(all_answers, modes_to_compare)
            highlighted_key_points = highlight_differences(all_key_points_text, modes_to_compare)
            
            # Create three columns for side-by-side comparison
            col1, col2, col3 = st.columns(3)
            
            columns = [col1, col2, col3]
            mode_names = ["Direct", "Planning", "Tool Calling"]
            
            for idx, (col, m, name) in enumerate(zip(columns, modes_to_compare, mode_names)):
                with col:
                    st.markdown(f"### {name} Mode")
                    result = mode_results[m]
                    
                    # Evidence Judge Results
                    st.markdown("**Evidence Judge**")
                    assessment = result.get('evidence_assessment', {})
                    st.write(f"Sufficient: {'✅' if assessment.get('is_sufficient') else '❌'}")
                    st.write(f"Coverage: {assessment.get('coverage_score', 0):.2f}")
                    st.write(f"Specificity: {assessment.get('specificity_score', 0):.2f}")
                    
                    # Retry count
                    st.markdown("**Retry Count**")
                    retry_count = result.get('retry_count', 0)
                    st.write(f"{retry_count} retries")
                    
                    # Final Answer
                    final_answer = result.get('final_answer', {})
                    if final_answer:
                        st.markdown("**Answer**")
                        if highlighted_answers[idx]:
                            with st.expander("Show Answer (differences highlighted)", expanded=False):
                                # Use highlighted version with differences marked
                                st.markdown(highlighted_answers[idx], unsafe_allow_html=True)
                        else:
                            st.write("_(No answer available)_")
                        
                        # Key Points
                        st.markdown("**Key Points**")
                        key_points = final_answer.get('key_points', [])
                        st.write(f"{len(key_points)} points")
                        if key_points:
                            with st.expander("Show Key Points (differences highlighted)", expanded=False):
                                # Use highlighted version
                                st.markdown(highlighted_key_points[idx], unsafe_allow_html=True)
                        
                        # Citations
                        st.markdown("**Citations**")
                        citation_map = final_answer.get('citation_map', {})
                        st.write(f"{len(citation_map)} citations")
                        with st.expander("Show Citations", expanded=False):
                            for num, chunk_info in citation_map.items():
                                if isinstance(chunk_info, dict):
                                    chunk_id = chunk_info.get('chunk_id', 'N/A')
                                else:
                                    chunk_id = str(chunk_info)
                                st.write(f"[{num}] {chunk_id}")
                        
                        # Confidence
                        st.markdown("**Confidence**")
                        st.write(f"{final_answer.get('confidence', 0):.2f}")
                    else:
                        st.write("No answer generated")

