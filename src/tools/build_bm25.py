# python ncci_rag/src/build_bm25.py
import argparse
import json
from bm25_store import BM25Store, tokenize
from rank_bm25 import BM25Okapi
import os

def main():
    base_dir = "ncci_rag/" if os.path.exists("ncci_rag/build") else ""
    ncci_chunks_path = f"{base_dir}build/chunks.jsonl"
    bm25_index_path = f"{base_dir}build/bm25_index.pkl"

    texts = []
    chunk_ids = []
    with open(ncci_chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_ids.append(c["chunk_id"])
            texts.append(c["text"])

    corpus = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(corpus)
    store = BM25Store(bm25=bm25, chunk_ids=chunk_ids)
    store.save(bm25_index_path)

    print(f"Built BM25 -> {bm25_index_path} (docs={len(chunk_ids)})")


if __name__ == "__main__":
    main()
