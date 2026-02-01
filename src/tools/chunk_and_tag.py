# python ncci_rag/src/chunk_and_tag.py
'''
Detect chapter/section/subsection, and generate chunks ('title + body' until next heading).
Extract CPT codes/ranges/modifiers from each chunk, and tag with topic tags.
'''
import argparse
import json
import os
import regex as re
from typing import Dict, Any, List, Optional
from regex_utils import extract_cpt_codes, extract_cpt_ranges, mentions_modifiers, extract_modifiers

CHAPTER_RE = re.compile(r"^\s*CHAPTER\s+([IVXLC]+)\b", re.IGNORECASE)
SECTION_RE = re.compile(r"^\s*([A-Z])\.\s+(.+?)\s*$")  # e.g., "E. Modifiers and Modifier Indicators"
SUBSECTION_RE = re.compile(r"^\s*([a-z])\.\s+(.+?)\s*$")  # e.g., "d. Modifier 59 ..."
ALLCAPS_RE = re.compile(r"^[A-Z0-9][A-Z0-9 \-–—,:;/()]{6,}$")


def normalize_line(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def classify_heading(line: str):
    line_n = normalize_line(line)
    if not line_n:
        return None

    m = CHAPTER_RE.match(line_n)
    if m:
        return ("chapter", f"CHAPTER {m.group(1).upper()}")

    m = SECTION_RE.match(line_n)
    if m:
        return ("section", f"{m.group(1).upper()}. {normalize_line(m.group(2))}")

    m = SUBSECTION_RE.match(line_n)
    if m:
        # Distinguish between subsection headings and list items
        # Use simple, reliable heuristics:
        letter = m.group(1)
        title = m.group(2)
        
        # List items (complete sentences) typically:
        # 1. End with a period
        # 2. Are long (>60 chars) without a colon
        # 3. Start with articles/determiners (The, A, An) - common in list items
        # 4. Single letters 'i' or 'v' might be roman numerals (part of i., ii., iii., iv., v.)
        
        # Subsection headings (noun phrases) typically:
        # 1. Short (<60 chars) or have a colon early
        # 2. Do NOT end with a period
        # 3. Are title-like (e.g., "Modifier 59", "Global Surgery Days")
        # 4. Do NOT start with articles (The, A, An)
        
        ends_with_period = title.rstrip().endswith('.')
        is_long_without_colon = len(title) > 60 and ':' not in title
        starts_with_article = title.startswith(('The ', 'A ', 'An '))
        
        # Roman numeral check: 'i' or 'v' likely part of roman numeral lists
        # Headings rarely use single 'i' or 'v' without being part of roman numerals
        # Exception: if followed by colon (e.g., "i. Introduction: ..."), it's a heading
        is_likely_roman_numeral = letter in ['i', 'v'] and ':' not in title
        
        # Strong indicators of list items
        if ends_with_period:
            # Sentences end with periods, headings don't
            return None
        
        if is_long_without_colon:
            # Long text without colon is likely a sentence, not a heading
            return None
        
        if starts_with_article:
            # Headings rarely start with "The", "A", "An"
            return None
        
        if is_likely_roman_numeral:
            # Single 'i' or 'v' without colon is likely part of roman numeral list
            return None
        
        # Otherwise, it's a subsection heading
        # If subsection title contains ':', only keep the part before ':'
        if ':' in title:
            title = title.split(':', 1)[0].strip()
        return ("subsection", f"{m.group(1)}. {normalize_line(title)}")

    # All-caps lines (often headings)
    if ALLCAPS_RE.match(line_n) and len(line_n.split()) <= 12:
        return ("caps", line_n)

    return None

def is_cpt_specific_chapter(chapter: str) -> bool:
    """
    Check if chapter contains CPT-specific coding guidelines.
    Chapters II-XIII contain code-range-specific policies.
    Introduction and Chapter I contain general/cross-cutting policies.
    """
    if not chapter:
        return False
    # Extract Roman numeral
    m = re.search(r'CHAPTER\s+([IVXLC]+)', chapter, re.IGNORECASE)
    if not m:
        return False
    roman = m.group(1).upper()
    # Chapter II onwards (2-13) are CPT-specific
    cpt_specific_romans = ['II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII']
    return roman in cpt_specific_romans

def classify_section_type(section: str) -> str:
    """
    Classify section into types based on common patterns across chapters.
    Returns: 'cross_cutting' | 'procedure_specific' | 'policy'
    """
    if not section:
        return "unknown"
    
    s = section.lower()
    
    # Cross-cutting sections that appear in most/all CPT chapters
    if "introduction" in s:
        return "cross_cutting"
    if "evaluation & management" in s or "e&m" in s:
        return "cross_cutting"
    if "anesthesia" in s and "anesthesia" not in s.replace("anesthesia", "", 1):  # Just section about anesthesia, not chapter II
        return "cross_cutting"
    if "medically unlikely edit" in s or "mue" in s:
        return "cross_cutting"
    if "general policy" in s:
        return "cross_cutting"
    
    # Procedure/service specific sections
    procedure_keywords = [
        "endoscop", "arthroscop", "laparoscop", "biopsy", "imaging",
        "lesion", "fracture", "repair", "graft", "injection",
        "cardiovascular", "respiratory", "digestive", "urinary",
        "nervous", "ophthalmology", "auditory", "spine", "breast",
        "laboratory", "pathology", "radiology", "nuclear medicine",
        "radiation oncology", "chemotherapy", "dialysis"
    ]
    if any(kw in s for kw in procedure_keywords):
        return "procedure_specific"
    
    # Policy sections
    if "ptp" in s or "ncci" in s or "edit" in s:
        return "policy"
    
    return "unknown"

def topic_tags_for(text: str, section: str, chapter: str) -> List[str]:
    t = text.lower()
    tags = set()

    # Content-based tags
    if "ptp" in t or "procedure-to-procedure" in t:
        tags.add("PTP")
    if "mue" in t or "medically unlikely edit" in t:
        tags.add("MUE")
    if "ccmi" in t or "modifier indicator" in t or "bypass" in t:
        tags.add("BYPASS")
    if "modifier" in t:
        tags.add("MODIFIER")
    if "global surgery" in t:
        tags.add("GLOBAL_SURGERY")
    if any(x in t for x in ["lt", "rt", "anatomic", "finger", "toe"]):
        tags.add("ANATOMIC")
    if "evaluation & management" in t or "e&m" in t:
        tags.add("E&M")
    if "general policy" in t:
        tags.add("GENERAL_POLICY")

    # Chapter-level tags
    if "introduction" in chapter.lower():
        tags.add("INTRODUCTION")
        tags.add("GENERAL_POLICY")
    elif chapter.startswith("CHAPTER I"):
        tags.add("CHAPTER_I")
        tags.add("GENERAL_POLICY")  # Chapter I is cross-cutting policy
    elif is_cpt_specific_chapter(chapter):
        tags.add("CPT_SPECIFIC")  # Chapters II-XIII are code-range specific

    # Section-based tags (more specific)
    s = (section or "").lower()
    if "modifier" in s:
        tags.add("MODIFIER")
    if "general policy" in s:
        tags.add("GENERAL_POLICY")
    if "ptp" in s:
        tags.add("PTP")
    if "mue" in s:
        tags.add("MUE")
    if "e&m" in s or "evaluation & management" in s:
        tags.add("E&M")
    if "anesthesia" in s:
        tags.add("ANESTHESIA")
    
    # Procedure-specific section tags
    if "endoscop" in s:
        tags.add("ENDOSCOPIC")
    if "arthroscop" in s:
        tags.add("ARTHROSCOPY")
    if "imaging" in s or "radiology" in s:
        tags.add("IMAGING")
    if "laboratory" in s or "pathology" in s:
        tags.add("LABORATORY")
    if "lesion" in s:
        tags.add("LESION")
    if "fracture" in s or "dislocation" in s:
        tags.add("FRACTURE")
    if "repair" in s or "reconstruction" in s:
        tags.add("REPAIR")
    if "cardiovascular" in s:
        tags.add("CARDIOVASCULAR")
    if "respiratory" in s:
        tags.add("RESPIRATORY")

    return sorted(tags)


def split_long_chunk(text: str, max_chars: int = 2000, overlap: int = 64) -> List[str]:
    """
    Split a long text into smaller chunks with overlap.
    Tries to split at sentence boundaries when possible.
    
    Args:
        text: The text to split
        max_chars: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks (approximately)
    
    Returns:
        List of text chunks
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Determine end position for this chunk
        end = start + max_chars
        
        if end >= len(text):
            # Last chunk
            chunks.append(text[start:])
            break
        
        # Try to find a sentence boundary (. ! ?) near the end
        # Look back up to 200 chars for a good break point
        search_start = max(end - 200, start)
        
        # Find all sentence endings in the search range
        sentence_endings = []
        for i in range(search_start, end):
            if text[i] in '.!?' and i + 1 < len(text):
                # Check if followed by space or newline (real sentence ending)
                if text[i + 1] in ' \n\t':
                    sentence_endings.append(i + 1)
        
        # Use the last sentence ending found, or just break at max_chars
        if sentence_endings:
            end = sentence_endings[-1]
        
        chunks.append(text[start:end].strip())
        
        # Move start position with overlap
        # Start next chunk at (end - overlap) to create overlap
        # But adjust to word boundary to avoid splitting words
        overlap_start = max(end - overlap, start + 1)
        
        # Find word boundary (space) near the overlap point
        # Search in a small window around overlap_start
        # Prefer searching backward first (smaller overlap), then forward
        next_start = overlap_start
        
        # Search backward up to 30 chars for a word boundary
        for i in range(overlap_start, max(overlap_start - 30, start), -1):
            if text[i] in ' \n\t':
                next_start = i + 1
                break
        else:
            # If not found backward, search forward up to 30 chars
            for i in range(overlap_start, min(overlap_start + 30, len(text))):
                if text[i] in ' \n\t':
                    next_start = i + 1
                    break
        
        start = next_start
    
    return chunks


def main():
    base_dir = "ncci_rag/" if os.path.exists("ncci_rag/build") else ""
    
    ncci_doc_path = f"{base_dir}build/pages.jsonl"
    ncci_output_path = f"{base_dir}build/chunks.jsonl"
    
    pages = []
    with open(ncci_doc_path, "r", encoding="utf-8") as f:
        for line in f:
            pages.append(json.loads(line))

    chunks_out = open(ncci_output_path, "w", encoding="utf-8")

    chapter = ""
    section = ""
    subsection = ""
    caps_heading = ""
    heading_path = []

    chunk_id = 0
    buffer_lines: List[str] = []
    buffer_page_start: Optional[int] = None
    buffer_page_end: Optional[int] = None

    def flush():
        nonlocal chunk_id, buffer_lines, buffer_page_start, buffer_page_end
        if not buffer_lines:
            return
        text = "\n".join(buffer_lines).strip()
        if len(text) < 250:
            # skip tiny chunks (TOC entries, chapter headers, incomplete fragments, etc.)
            # 250 chars ≈ 35-40 words ≈ 2-3 complete sentences
            buffer_lines = []
            buffer_page_start = None
            buffer_page_end = None
            return

        # Determine content type based on chapter
        content_type = "general_policy"  # default
        if "introduction" in chapter.lower():
            content_type = "introduction"
        elif chapter.startswith("CHAPTER I"):
            content_type = "general_policy"
        elif is_cpt_specific_chapter(chapter):
            content_type = "cpt_specific"
        
        # Classify section type
        section_type = classify_section_type(section)
        
        # Split long chunks if needed
        text_chunks = split_long_chunk(text, max_chars=2000, overlap=64)
        
        # Record the logical chunk number (before splitting) for grouping
        logical_chunk_num = chunk_id + 1
        
        # Create a chunk record for each sub-chunk
        for sub_idx, sub_text in enumerate(text_chunks):
            chunk_id += 1  # Each sub-chunk gets its own sequential ID
            
            # Extract metadata from this sub-chunk
            codes = extract_cpt_codes(sub_text)
            ranges = extract_cpt_ranges(sub_text)
            mods = extract_modifiers(sub_text)
            
            rec = {
                "chunk_id": f"chunk_{chunk_id:06d}",  # Simple sequential ID
                "text": sub_text,
                "page_start": buffer_page_start,
                "page_end": buffer_page_end,
                "chapter": chapter,
                "section": section,
                "subsection": subsection,
                "caps_heading": caps_heading,
                "heading_path": [h for h in [chapter, section, subsection, caps_heading] if h],
                "content_type": content_type,  # introduction | general_policy | cpt_specific
                "section_type": section_type,  # cross_cutting | procedure_specific | policy | unknown
                "cpt_codes": codes,
                "cpt_ranges": ranges,
                "mentions_modifiers": mentions_modifiers(sub_text),
                "modifiers_found": mods,
                "topic_tags": topic_tags_for(sub_text, section, chapter),
                "is_split": len(text_chunks) > 1,  # Flag to indicate if this was split
                "split_index": sub_idx + 1 if len(text_chunks) > 1 else None,  # Which part (1, 2, 3, etc.)
                "total_splits": len(text_chunks) if len(text_chunks) > 1 else None,  # Total number of splits
                "parent_chunk": logical_chunk_num if len(text_chunks) > 1 else None,  # Group related splits
            }
            chunks_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

        buffer_lines = []
        buffer_page_start = None
        buffer_page_end = None

    for p in pages:
        page_no = p["page_no"]
        lines = p["text"].splitlines()

        for raw in lines:
            line = normalize_line(raw)
            if not line:
                continue
            
            # Skip page footer lines (Revision Date and page numbers like "I-20")
            if 'Revision Date' in line or (line.startswith('I-') and len(line) < 10):
                continue

            h = classify_heading(line)
            if h:
                # when encountering new heading, flush current buffer
                flush()
                kind, value = h
                if kind == "chapter":
                    chapter = value
                    section = ""
                    subsection = ""
                    caps_heading = ""
                elif kind == "section":
                    section = value
                    subsection = ""
                    caps_heading = ""
                elif kind == "subsection":
                    subsection = value
                    caps_heading = ""
                    # Check if there's content after the subsection heading on the same line
                    # e.g., "e. Modifiers XE, XS, XP, XU: These modifiers were effective..."
                    subsection_match = SUBSECTION_RE.match(line)
                    if subsection_match:
                        # Extract full title part (before splitting at ':')
                        full_title = subsection_match.group(2)
                        # Check if there's a colon - content after ':' becomes text
                        if ':' in full_title:
                            parts = full_title.split(':', 1)
                            remaining = parts[1].strip() if len(parts) > 1 else ""
                            if remaining and len(remaining) > 10:  # Has meaningful content
                                if buffer_page_start is None:
                                    buffer_page_start = page_no
                                buffer_page_end = page_no
                                buffer_lines.append(remaining)
                        else:
                            # No colon, check if there's content after the title
                            remaining = line[subsection_match.end():].strip()
                            if remaining and len(remaining) > 10:  # Has meaningful content
                                if buffer_page_start is None:
                                    buffer_page_start = page_no
                                buffer_page_end = page_no
                                buffer_lines.append(remaining)
                    continue
                elif kind == "caps":
                    caps_heading = value
                continue

            # normal content line
            if buffer_page_start is None:
                buffer_page_start = page_no
            buffer_page_end = page_no
            buffer_lines.append(line)

        # page boundary: keep buffering (do not flush)

    flush()
    chunks_out.close()
    print(f"Wrote chunks -> {ncci_output_path}")


if __name__ == "__main__":
    main()
