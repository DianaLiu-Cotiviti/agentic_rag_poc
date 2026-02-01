import pickle
from dataclasses import dataclass
from typing import List, Dict, Any
import regex as re
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [t.group(0).lower() for t in TOKEN_RE.finditer(text)]


@dataclass
class BM25Store:
    bm25: BM25Okapi
    chunk_ids: List[str]

    def search(self, query: str, top_k: int = 20):
        q = tokenize(query)
        scores = self.bm25.get_scores(q)
        # top-k indices
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"chunk_id": self.chunk_ids[i], "score": float(scores[i])} for i in idxs]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunk_ids": self.chunk_ids}, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return cls(bm25=obj["bm25"], chunk_ids=obj["chunk_ids"])
