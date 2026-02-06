# python ncci_rag/src/build_range_index.py
import json
import os
import sqlite3
from typing import Dict, Any, List


DDL = """
CREATE TABLE IF NOT EXISTS range_index (
  start INTEGER NOT NULL,
  end   INTEGER NOT NULL,
  chunk_id TEXT NOT NULL,
  chapter TEXT,
  section TEXT,
  page_start INTEGER,
  page_end INTEGER,
  content_type TEXT,
  section_type TEXT,
  weight REAL DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_range_start_end ON range_index(start, end);
CREATE INDEX IF NOT EXISTS idx_range_chunk ON range_index(chunk_id);
"""


def weight_for(chapter: str, section: str, tags: List[str], 
               content_type: str = "unknown", section_type: str = "unknown") -> float:
    """
    Calculate retrieval weight based on chunk metadata.
    
    Weight strategy (higher = more relevant):
    
    Layer 1 - Content Type:
      * cpt_specific: +2.5 (Content about specific CPT range - most relevant!)
      * general_policy: +0.3 (General policies applicable to all codes - secondary)
      * introduction: +0.1 (Background information - least relevant)
    
    Layer 2 - Section Type:
      * procedure_specific: +2.0 (Procedure-specific rules, most relevant)
      * cross_cutting: +0.8 (Cross-chapter rules like E&M, MUE)
      * policy: +0.6 (NCCI edit policies)
    
    Layer 3 - Topic Tags:
      * MODIFIER/BYPASS: +1.2 (Modifier rules - very important)
      * PTP: +1.0 (PTP edits - very important)
      * CHAPTER_I: +0.4 (General foundational rules)
      * MUE: +0.3 (MUE values)
    
    Total weight range: 1.0 (baseline) to ~6.0+ (highly relevant)
    """
    w = 1.0  # Baseline weight
    
    # Layer 1: Content type weighting
    if content_type == "cpt_specific":
        w += 2.5  # Most relevant: sections about specific CPT
    elif content_type == "general_policy":
        w += 0.3  # Secondary: general policies apply but less specific
    elif content_type == "introduction":
        w += 0.1  # Least relevant: background information
    
    # Layer 2: Section type weighting
    if section_type == "procedure_specific":
        w += 2.0  # Most relevant: coding rules for specific procedures
    elif section_type == "cross_cutting":
        w += 0.8  # Very relevant: E&M, MUE and other general sections
    elif section_type == "policy":
        w += 0.6  # Relevant: NCCI edit policies
    
    # Layer 3: Topic tags weighting
    if "MODIFIER" in tags or "BYPASS" in tags:
        w += 1.2  # Very important: modifier usage rules
    if "PTP" in tags:
        w += 1.0  # Very important: PTP edit rules
    if "CHAPTER_I" in tags:
        w += 0.4  # General foundational rules
    if "MUE" in tags:
        w += 0.3  # MUE values
    
    return w


def build_range_db_index(chunks_path: str, index_path: str):
    """
    构建Range Index
    
    Args:
        chunks_path: chunks.jsonl文件路径
        index_path: 输出的数据库路径
    """
    conn = sqlite3.connect(index_path)
    cur = conn.cursor()
    for stmt in DDL.strip().split(";"):
        s = stmt.strip()
        if s:
            cur.execute(s)

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            cid = c["chunk_id"]
            chapter = c.get("chapter", "")
            section = c.get("section", "")
            ps = c.get("page_start")
            pe = c.get("page_end")
            tags = c.get("topic_tags", [])
            content_type = c.get("content_type", "unknown")
            section_type = c.get("section_type", "unknown")

            w = weight_for(chapter, section, tags, content_type, section_type)

            # single codes as (code, code)
            for code in c.get("cpt_codes", []):
                cur.execute(
                    "INSERT INTO range_index(start,end,chunk_id,chapter,section,page_start,page_end,content_type,section_type,weight) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (code, code, cid, chapter, section, ps, pe, content_type, section_type, w),
                )

            # ranges
            for r in c.get("cpt_ranges", []):
                cur.execute(
                    "INSERT INTO range_index(start,end,chunk_id,chapter,section,page_start,page_end,content_type,section_type,weight) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (r["start"], r["end"], cid, chapter, section, ps, pe, content_type, section_type, w),
                )

    conn.commit()
    conn.close()
    print(f"Built range index -> {index_path}")


def main():
    from ..config import AgenticRAGConfig
    
    config = AgenticRAGConfig.from_env()
    build_range_db_index(config.chunks_path, config.range_index_path)


if __name__ == "__main__":
    main()
