# python ncci_rag/src/retrieve.py
'''
Docstring for ncci_rag.src.06_retrieve
Range routing + policy expansion + hybrid(BM25+Semantic RRF) + evidence output

Retrieval Pipeline:
1. Range routing: SQLite range index lookup (CPT code routing)
2. Policy expansion: Build enhanced query with modifiers/tags
3. Hybrid retrieval: BM25 (lexical) + Semantic (embedding) with RRF fusion
'''
import argparse
import json
import os
import sqlite3
from typing import Dict, List, Set, Tuple

from openai import AzureOpenAI
from .bm25_store import BM25Store
from .chroma_store import ChromaStore
from dotenv import load_dotenv

load_dotenv()


def load_chunks_map(chunks_path: str) -> Dict[str, dict]:
    m = {}
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            m[c["chunk_id"]] = c
    return m


def range_lookup(range_db: str, cpt: int, limit: int = 300) -> List[Tuple[str, float]]:
    conn = sqlite3.connect(range_db)
    cur = conn.cursor()
    cur.execute(
        "SELECT chunk_id, weight FROM range_index WHERE start <= ? AND end >= ? ORDER BY weight DESC LIMIT ?",
        (cpt, cpt, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return [(r[0], float(r[1])) for r in rows]


def build_policy_query(cpt: int) -> str:
    """Build BM25 keyword query (for policy expansion)"""
    return (
        f"{cpt} modifier PTP bypass CCMI modifier indicator "
        f"59 XE XP XS XU LT RT anatomic global surgery distinct separate encounter session"
    )


def build_semantic_query(cpt: int) -> str:
    """Build semantic query prompt (for semantic search)
    
    Design principles:
    1. Clear query objective (CPT code + related policies)
    2. Cover key aspects (edit rules, modifiers, bundling, global/components, etc.)
    3. Professional terminology (PTP edits, CCMI, bypass indicators)
    4. Completeness (general policies + code-specific rules)
    """
    return (
        f"Find all relevant NCCI documentation for CPT code {cpt}, including:\n"
        f"1. PTP (Procedure-to-Procedure) edits and bundling rules for {cpt}\n"
        f"2. Modifier usage policies (59, XE, XP, XS, XU, anatomic modifiers LT/RT)\n"
        f"3. CCMI (Column 1/Column 2 Correct Coding Modifier Indicator) and bypass indicators\n"
        f"4. MUE (Medically Unlikely Edits) policies if applicable\n"
        f"5. General policies from Chapter I that apply to this procedure\n"
        f"6. Global surgery packages, anatomic considerations, and distinct procedural services\n"
        f"7. Add-on codes, bilateral procedures, and separate encounter documentation requirements"
    )


def embed_query(client: AzureOpenAI, deployment: str, text: str):
    resp = client.embeddings.create(model=deployment, input=text)
    return resp.data[0].embedding


def rrf_fuse(*ranked_lists: List[dict], k: int = 60) -> Dict[str, float]:
    # Reciprocal Rank Fusion
    scores: Dict[str, float] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst, start=1):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


def multi_stage_hybrid_rag(target_cpt_code: int = None, top_k: int = 15):
    """Multi-stage hybrid RAG retrieval
    
    Args:
        target_cpt_code: Target CPT code
        top_k: Number of top-k results to return (default: 10)
    """
    base_dir = "ncci_rag/" if os.path.exists("ncci_rag/build") else ""
    ncci_chunks_path = f"{base_dir}build/chunks.jsonl"
    cpt_range_index_path = f"{base_dir}build/cpt_range_index.db"
    bm25_index_path = f"{base_dir}build/bm25_index.pkl"
    chroma_db_path = f"{base_dir}build/chroma_db"
    
    # Prompt user input if CPT code not provided
    if target_cpt_code is None:
        try:
            target_cpt_code = int(input("Please enter target CPT code: "))
        except ValueError:
            print("❌ Invalid CPT code. Please enter a valid integer.")
            return None
    
    print(f"\n{'='*60}")
    print(f"🔍 Multi-Stage Hybrid RAG Retrieval for CPT {target_cpt_code}")
    print(f"{'='*60}")

    bm25 = BM25Store.load(bm25_index_path)
    chroma_store = ChromaStore(chroma_db_path, "ncci_chunks")
    chunks_map = load_chunks_map(ncci_chunks_path)

    client = AzureOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY_EMBEDDING", os.environ["AZURE_OPENAI_API_KEY"]),
        api_version=os.environ["AZURE_API_VERSION_EMBEDDING"],
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT_EMBEDDING", os.environ["AZURE_OPENAI_ENDPOINT"]),
    )
    emb_deploy = os.environ["AZURE_DEPLOYMENT_NAME_EMBEDDING"]

    # ===== STAGE A: Range Routing =====
    print("\n=== STAGE A: Range Routing (CPT Code Lookup) ===")
    routed = range_lookup(cpt_range_index_path, target_cpt_code)
    C_code = set(cid for cid, _ in routed)
    print(f"Found {len(C_code)} chunks containing CPT {target_cpt_code}")

    # ===== STAGE B: Policy Expansion =====
    # Use Semantic Search to find general policies (avoid duplication with STAGE C BM25)
    # Advantage: Semantic understanding can find policy content without exact keyword matches
    print("\n=== STAGE B: Policy Expansion (Semantic Search for policies) ===")
    
    # Build dedicated policy query
    policy_semantic_query = (
        f"General NCCI policies, modifier usage guidelines, PTP edit rules, "
        f"bypass indicators, CCMI policies, Chapter I general policies that apply to CPT {target_cpt_code}"
    )
    print(f"Policy query: {policy_semantic_query[:50]}...")
    
    # Use Semantic Search to find policy chunks (limit to 40)
    policy_emb = embed_query(client, emb_deploy, policy_semantic_query)
    policy_hits = chroma_store.search(policy_emb, top_k=40)
    
    C_policy = set()
    for item in policy_hits:
        c = chunks_map.get(item["chunk_id"])
        if not c:
            continue
        tags = set(c.get("topic_tags", []))
        # Only keep true policy chunks (confirmed by tags)
        if "MODIFIER" in tags or "BYPASS" in tags or "GENERAL_POLICY" in tags or "CHAPTER_I" in tags or "PTP" in tags:
            C_policy.add(item["chunk_id"])
            if len(C_policy) >= 40:  # Limit to 40 policy chunks
                break
    print(f"Added {len(C_policy)} policy-related chunks (semantic search + tag filtering, max 40)")

    # Combine routing + policy
    candidate_set = C_code.union(C_policy)
    print(f"Total candidate set: {len(candidate_set)} chunks")
    
    if not candidate_set:
        print("⚠ No candidates found. Falling back to global search.")
        candidate_set = set(chunks_map.keys())

    # ===== STAGE C: Hybrid Retrieval on Policy Chunks Only =====
    # Priority: Keep ALL range routing chunks, then fill remaining slots with best policy chunks
    print("\n=== STAGE C: Hybrid Retrieval (BM25 + Semantic on policy chunks) ===")
    
    # Calculate how many policy chunks we need
    num_code_chunks = len(C_code)
    remaining_slots = max(0, top_k - num_code_chunks)
    print(f"Range routing chunks: {num_code_chunks} (kept all)")
    print(f"Policy slots available: {remaining_slots}")
    
    if remaining_slots > 0:
        # Build queries
        keyword_query = build_policy_query(target_cpt_code)  # BM25 keyword query
        semantic_query = build_semantic_query(target_cpt_code)  # Semantic full query
        
        # Only search in policy chunks (exclude code chunks to avoid duplication)
        policy_only = C_policy - C_code
        print(f"Hybrid search on {len(policy_only)} policy chunks")
        
        # BM25: lexical keyword search (within policy chunks only)
        bm25_hits = bm25.search(keyword_query, top_k=100)
        bm25_hits = [h for h in bm25_hits if h["chunk_id"] in policy_only][:30]
        print(f"  - BM25 (lexical) hits: {len(bm25_hits)}")

        # ChromaDB: semantic search (within policy chunks only)
        query_emb = embed_query(client, emb_deploy, semantic_query)
        chroma_hits = chroma_store.search(query_emb, top_k=100)
        chroma_hits = [h for h in chroma_hits if h["chunk_id"] in policy_only][:30]
        print(f"  - Semantic hits: {len(chroma_hits)}")

        # ===== STAGE D: RRF Fusion for Policy Chunks =====
        print("\n=== STAGE D: RRF Fusion (policy chunks only) ===")
        fused_policy = rrf_fuse(bm25_hits, chroma_hits, k=60)
        top_policy = sorted(fused_policy.items(), key=lambda x: x[1], reverse=True)[:remaining_slots]
        print(f"Selected top-{len(top_policy)} policy chunks")
        
        # Combine: All code chunks + top policy chunks
        final_chunks = []
        # Add all range routing chunks with high priority scores
        for cid in C_code:
            final_chunks.append((cid, 1.0))  # Max score for code chunks
        # Add selected policy chunks
        for cid, score in top_policy:
            final_chunks.append((cid, score * 0.5))  # Lower score for policy chunks
        
        # Sort by score (code chunks will be first)
        top = sorted(final_chunks, key=lambda x: x[1], reverse=True)[:top_k]
    else:
        # If we have too many code chunks, just take top_k of them
        print("\n=== STAGE D: Using range routing chunks only ===")
        top = [(cid, 1.0) for cid in list(C_code)[:top_k]]
    
    print(f"Final result: {len(top)} chunks total")

    # Print results with evidence
    print(f"\n=== TOP {len(top)} evidence chunks for CPT {target_cpt_code} ===\n")
    results = []
    for rank, (cid, score) in enumerate(top, start=1):
        c = chunks_map[cid]
        # Determine source (range routing or policy expansion)
        source = "range_routing" if cid in C_code else "policy_expansion"
        
        # Save complete information for subsequent LLM calls
        rec = {
            "rank": rank,
            "chunk_id": cid,
            "source": source,  # New field to track chunk origin
            "rrf_score": float(score),
            "full_text": c["text"],  # Full text (needed by LLM)
            "pages": [c.get("page_start"), c.get("page_end")],
            "chapter": c.get("chapter"),
            "section": c.get("section"),
            "heading_path": c.get("heading_path"),
            "topic_tags": c.get("topic_tags"),
            "content_type": c.get("content_type"),
            "section_type": c.get("section_type")
        }
        results.append(rec)

    # Save to output directory for subsequent LLM calls
    output_dir = f"{base_dir}output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/retrieved_chunks_cpt_{target_cpt_code}.json"
    
    # Count chunks by source
    num_routing_chunks = sum(1 for r in results if r["source"] == "range_routing")
    num_policy_chunks = sum(1 for r in results if r["source"] == "policy_expansion")
    
    output_data = {
        "cpt_code": target_cpt_code,
        "top_k": len(results),
        "retrieval_strategy": "prioritize_range_routing",
        "total_candidates": len(candidate_set),
        "retrieval_stages": {
            "range_routing_found": len(C_code),
            "range_routing_kept": num_routing_chunks,
            "policy_expansion_candidates": len(C_policy),
            "policy_expansion_selected": num_policy_chunks,
            "bm25_hits": len(bm25_hits) if remaining_slots > 0 else 0,
            "semantic_hits": len(chroma_hits) if remaining_slots > 0 else 0
        },
        "chunks": results
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved {len(results)} chunks to: {output_path}")
    print(f"   🎯 Range routing: {num_routing_chunks} chunks")
    print(f"   📋 Policy expansion: {num_policy_chunks} chunks")
    print(f"   Ready for LLM response generation!\n")
    
    return results


# if __name__ == "__main__":
#     multi_stage_hybrid_rag(target_cpt_code = 97810)
