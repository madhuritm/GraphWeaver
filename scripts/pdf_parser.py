#!/usr/bin/env python3
import re, json, hashlib, argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import pdfplumber
import tiktoken

@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    page: int
    text: str
    char_start: int
    char_end: int
    n_tokens: int
    text_hash: str

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r", "\n")
    # common unicode normalizations (optional, helps offsets be stable)
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = s.replace("–", "-").replace("—", "-")
    # de-hyphenate at line breaks, collapse whitespace, preserve blank lines as paragraph breaks
    s = re.sub(r"-\n(?=\w)", "", s)
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)
    s = re.sub(r"\n{2,}", "\n\n", s)
    return s.strip()

def chunk_page_text(page_text_raw: str, page_num: int, doc_id: str, enc, max_tokens: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    if not page_text_raw or not page_text_raw.strip():
        return chunks

    ##normalize_text cleans up punctuaton, whitespace, lowercase, unicode normalization
    page_text = normalize_text(page_text_raw)

    # paragraph-aware: prefer splitting on blank lines; fall back to single newlines
    paragraphs = page_text.split("\n\n") if "\n\n" in page_text else page_text.split("\n")

    buf, cur_tokens = [], 0

    def flush():
        nonlocal buf, cur_tokens
        if not buf:
            return
        chunk_text = " ".join(x.strip() for x in buf if x.strip())
        token_ids = enc.encode(chunk_text)
        n_tokens = len(token_ids)

        # best-effort char span: search from last end to reduce ambiguity
        start_from = chunks[-1].char_end if chunks else 0
        idx = page_text.find(chunk_text, start_from)
        if idx < 0:
            idx = page_text.find(chunk_text)
        char_start = max(idx, 0)
        char_end = char_start + len(chunk_text)

        key = f"{doc_id}|p{page_num}|{char_start}-{char_end}"
        chunk_id = hashlib.sha1(key.encode()).hexdigest()

        chunks.append(Chunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            page=page_num,
            text=chunk_text,
            char_start=char_start,
            char_end=char_end,
            n_tokens=n_tokens,
            text_hash=hashlib.sha1(chunk_text.encode()).hexdigest()
        ))
        buf, cur_tokens = [], 0

    for para in paragraphs:
        p = para.strip()
        if not p:
            continue
        p_tokens = enc.encode(p)
        p_len = len(p_tokens)

        if cur_tokens > 0 and cur_tokens + p_len > max_tokens:
            flush()
        buf.append(p)
        cur_tokens += p_len

        if cur_tokens >= max_tokens:
            flush()

    flush()
    return chunks

def parse_pdf_to_chunks(path: str, doc_id: str, max_tokens: int = 1000) -> List[Dict[str, Any]]:
    """Reads a text-based PDF and returns a list of chunk dicts (in-memory)."""
    enc = tiktoken.get_encoding("cl100k_base")
    out: List[Dict[str, Any]] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            print(text)
            out.extend(asdict(c) for c in chunk_page_text(text, i, doc_id, enc, max_tokens))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to a text-based PDF")
    ap.add_argument("--doc-id", required=True, help="Stable ID you assign this document")
    ap.add_argument("--max-tokens", type=int, default=1000, help="Target tokens per chunk")
    ap.add_argument("--out", help="Optional path to write JSON; otherwise prints to stdout")
    args = ap.parse_args()

    chunks = parse_pdf_to_chunks(args.pdf, args.doc_id, args.max_tokens)  # kept in memory here

    payload = json.dumps(chunks, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload)
    else:
        print(payload)

if __name__ == "__main__":
    main()
