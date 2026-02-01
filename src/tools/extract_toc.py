# python ncci_rag/src/extract_toc.py
"""
Extract the Table of Contents from NCCI Manual pages.jsonl
This helps understand the structure of each chapter before chunking.
"""
import json
import regex as re
from typing import List, Dict, Any

CHAPTER_RE = re.compile(r"^\s*Chapter\s+([IVXLC]+)\s*[–—-]\s*(.+?)$", re.IGNORECASE)
CHAPTER_CPT_RANGE_RE = re.compile(r"\(CPT\s+[Cc]odes?\s+(\d{5})\s*[–—-]\s*(\d{5})\)", re.IGNORECASE)
SECTION_LETTER_RE = re.compile(r"^\s*([A-Z])\.\s+(.+?)\s*$")  # Just section letter and name
PAGE_REF_RE = re.compile(r"^([IVX]+-\d+|Intro-\d+)\s*$")  # Page reference on next line
INTRO_SECTION_RE = re.compile(r"^(.+?)\s+(Intro-\d+)\s*$")


def extract_toc(pages_jsonl: str, output_json: str):
    """
    Extract Table of Contents structure from pages.jsonl
    """
    toc = {
        "introduction": [],
        "chapters": []
    }
    
    current_chapter = None
    in_toc = False
    pending_section = None  # Store section waiting for page reference
    
    with open(pages_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            page = json.loads(line)
            page_no = page['page_no']
            text = page['text']
            
            # Stop when we reach the actual content (after TOC pages)
            if page_no > 15:  # TOC is typically in first ~10 pages
                break
            
            # Look for "Table of Contents" marker
            if "Table of Contents" in text or "CHAPTER I" in text:
                in_toc = True
            
            if not in_toc:
                continue
            
            lines = text.splitlines()
            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    continue
                
                # Check if this is a page reference for pending section
                if pending_section:
                    page_match = PAGE_REF_RE.match(line)
                    if page_match:
                        pending_section["page_ref"] = page_match.group(1)
                        if current_chapter:
                            current_chapter["sections"].append(pending_section)
                        pending_section = None
                        continue
                
                # Match Introduction sections (some have page on same line)
                intro_match = INTRO_SECTION_RE.match(line)
                if intro_match:
                    section_name = intro_match.group(1).strip()
                    page_ref = intro_match.group(2).strip()
                    if "Introduction" not in section_name or len(section_name) > 10:
                        toc["introduction"].append({
                            "section": section_name,
                            "page_ref": page_ref
                        })
                    continue
                
                # Match chapter headers
                chapter_match = CHAPTER_RE.match(line)
                if chapter_match:
                    pending_section = None  # Clear any pending
                    roman = chapter_match.group(1).upper()
                    title = chapter_match.group(2).strip()
                    
                    current_chapter = {
                        "chapter_number": roman,
                        "title": title,
                        "cpt_range": None,
                        "sections": []
                    }
                    toc["chapters"].append(current_chapter)
                    continue
                
                # Extract CPT code range (usually on next line after chapter)
                if current_chapter and not current_chapter["cpt_range"]:
                    cpt_match = CHAPTER_CPT_RANGE_RE.search(line)
                    if cpt_match:
                        current_chapter["cpt_range"] = {
                            "start": cpt_match.group(1),
                            "end": cpt_match.group(2)
                        }
                        continue
                
                # Match section entries (A. Section Name, page ref on next line)
                if current_chapter:
                    section_match = SECTION_LETTER_RE.match(line)
                    if section_match:
                        letter = section_match.group(1)
                        section_name = section_match.group(2).strip()
                        
                        pending_section = {
                            "letter": letter,
                            "name": section_name,
                            "page_ref": None
                        }
                        # Don't append yet, wait for page reference
    
    # Save TOC to JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(toc, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Extracted TOC with {len(toc['chapters'])} chapters")
    print(f"✓ Saved to {output_json}")
    
    # Print summary
    print("\n" + "="*60)
    print("TABLE OF CONTENTS SUMMARY")
    print("="*60)
    
    print(f"\nIntroduction ({len(toc['introduction'])} sections):")
    for item in toc['introduction'][:5]:
        print(f"  • {item['section']} [{item['page_ref']}]")
    
    print(f"\nChapters ({len(toc['chapters'])}):")
    for ch in toc['chapters']:
        cpt_info = ""
        if ch['cpt_range']:
            cpt_info = f" (CPT {ch['cpt_range']['start']}-{ch['cpt_range']['end']})"
        print(f"\n  Chapter {ch['chapter_number']}: {ch['title']}{cpt_info}")
        print(f"    └─ {len(ch['sections'])} sections:")
        for sec in ch['sections'][:3]:
            print(f"       {sec['letter']}. {sec['name']} [{sec['page_ref']}]")
        if len(ch['sections']) > 3:
            print(f"       ... and {len(ch['sections']) - 3} more")
    
    return toc


def main():
    import os
    base_dir = "ncci_rag/" if os.path.exists("ncci_rag/build") else ""
    
    pages_jsonl = f"{base_dir}build/pages.jsonl"
    output_json = f"{base_dir}build/table_of_contents.json"
    
    if not os.path.exists(pages_jsonl):
        print(f"Error: {pages_jsonl} not found")
        print("Please run 01_extract_pdf.py first")
        return
    
    extract_toc(pages_jsonl, output_json)


if __name__ == "__main__":
    main()
