# Usage (from project root): python ncci_rag/src/extract_pdf.py
"""
Step 1: Extract PDF to JSONL
Supports execution from both project root and ncci_rag directory
"""
import argparse
import json
import os
import fitz  # PyMuPDF


def main():
    # Path adaptation: support execution from project root or ncci_rag directory
    base_dir = "ncci_rag/" if os.path.exists("ncci_rag/data") else ""
    
    ncci_doc_path = f"{base_dir}data/ncci_manual.pdf"
    output_path = f"{base_dir}build/pages.jsonl"
    os.makedirs(f"{base_dir}build", exist_ok=True)
    
    doc = fitz.open(ncci_doc_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text("text")
            rec = {"page_no": i + 1, "text": text}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(doc)} pages -> {output_path}")


if __name__ == "__main__":
    main()
