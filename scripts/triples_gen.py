#!/usr/bin/env python3
import os, json, time, argparse, hashlib, re
from collections import defaultdict
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from openai import OpenAI

# ----------------------------
# Config
# ----------------------------
MODEL = os.getenv("OPENAI_TRIPLES_MODEL", "gpt-5")
TEMPERATURE = float(os.getenv("OPENAI_TRIPLES_TEMPERATURE", "0"))
MAX_RETRIES = int(os.getenv("OPENAI_TRIPLES_MAX_RETRIES", "3"))
RETRY_BACKOFF = float(os.getenv("OPENAI_TRIPLES_RETRY_BACKOFF", "2.0"))

SYSTEM_INSTRUCTIONS = (
    "You are an expert relation extraction system.\n"
    "Given TEXT and an indexed list of ENTITIES (index, text, label), extract explicit subject–predicate–object triples.\n"
    "RULES:\n"
    "1) Subjects/objects MUST be chosen ONLY from the provided ENTITIES by their indices.\n"
    "2) Predicate must be a SHORT relation key that reflects the phrasing in the text (e.g., 'treats', 'located_in', 'reports').\n"
    "3) Keep predicate concise: ≤ 3 words, no punctuation, prefer a single verb or verb_phrase.\n"
    "4) If nothing clear is stated, return an empty list.\n"
    "5) Optionally include a confidence in [0,1].\n"
    "Return JSON only per schema."
)

# Expected model output:
# { "triples": [ { "subject_idx": int, "predicate": str, "object_idx": int, "confidence": float? } ] }

@dataclass
class Entity:
    idx: int
    text: str
    label: str
    start: int
    end: int
    doc_char_start: int
    doc_char_end: int

@dataclass
class TripleOut:
    doc_id: str
    chunk_id: str
    page: int
    subject: str
    subject_label: str
    subject_start: int
    subject_end: int
    subject_doc_char_start: int
    subject_doc_char_end: int
    predicate: str
    object: str
    object_label: str
    object_start: int
    object_end: int
    object_doc_char_start: int
    object_doc_char_end: int
    triple_id: str
    confidence: Optional[float] = None

# ----------------------------
# Helpers
# ----------------------------
def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _snake(s: str) -> str:
    s = s.strip().lower()
    # keep only letters/numbers/spaces/underscores
    s = re.sub(r"[^a-z0-9 _]", "", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # limit to 3 words
    parts = s.split(" ")
    parts = parts[:3]
    s = "_".join([p for p in parts if p])
    # length cap
    return s[:48] if len(s) > 48 else s

def load_predicate_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # normalize keys/values to snake_case
    out = {}
    for k, v in data.items():
        out[_snake(str(k))] = _snake(str(v))
    return out

def normalize_predicate(raw: str, pred_map: Dict[str, str]) -> str:
    if not raw:
        return ""
    norm = _snake(raw)
    if not norm:
        return ""
    # optional remap
    return pred_map.get(norm, norm)

def load_chunks(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return {c["chunk_id"]: c for c in chunks}

def load_ner_grouped(path: str) -> Dict[str, List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        ents = json.load(f)
    by_chunk: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in ents:
        by_chunk[e["chunk_id"]].append(e)
    for k in by_chunk:
        by_chunk[k].sort(key=lambda x: (x.get("start", 0), x.get("end", 0), x.get("label", ""), x.get("text", "")))
    return by_chunk

def build_entities_for_chunk(ner_items: List[Dict[str, Any]]) -> List[Entity]:
    out: List[Entity] = []
    for idx, e in enumerate(ner_items):
        out.append(Entity(
            idx=idx,
            text=e["text"],
            label=e.get("label", "ENTITY"),
            start=int(e.get("start", 0)),
            end=int(e.get("end", 0)),
            doc_char_start=int(e.get("doc_char_start", 0)),
            doc_char_end=int(e.get("doc_char_end", 0)),
        ))
    return out

def format_entities_for_prompt(entities: List[Entity]) -> str:
    return "\n".join(f'[{e.idx}] "{e.text}" ({e.label}) span={e.start}-{e.end}' for e in entities)

def dedupe_triples(rows: List[TripleOut]) -> List[TripleOut]:
    seen = set()
    out = []
    for r in rows:
        key = (r.doc_id, r.chunk_id, r.subject, r.predicate, r.object)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

# ----------------------------
# OpenAI call (chat.completions)
# ----------------------------
def call_openai_triples(client: OpenAI, text: str, entity_list_for_prompt: str) -> Dict[str, Any]:
    user_prompt = (
        "Extract subject–predicate–object triples using ONLY the indexed ENTITIES below.\n\n"
        f"ENTITIES:\n{entity_list_for_prompt}\n\n"
        "Output JSON ONLY in this exact schema:\n"
        '{ "triples": [ {"subject_idx": int, "predicate": string, "object_idx": int, "confidence": float?} ] }\n\n'
        "Keep predicate a short relation key (≤3 words, no punctuation). If unclear, return [].\n\n"
        f"TEXT:\n{text}\n"
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                #temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content if resp.choices else ""
            if not content:
                raise ValueError("Empty content")
            return json.loads(content)
        except Exception:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_BACKOFF * attempt)

# ----------------------------
# Main extraction
# ----------------------------
def extract_triples(chunks_by_id: Dict[str, Dict[str, Any]],
                    ner_by_chunk: Dict[str, List[Dict[str, Any]]],
                    pred_map: Dict[str, str]) -> List[Dict[str, Any]]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    results: List[TripleOut] = []

    for chunk_id, ner_items in ner_by_chunk.items():
        ch = chunks_by_id.get(chunk_id)
        if not ch:
            continue
        text = ch["text"]
        doc_id = ch["doc_id"]
        page = ch.get("page", 0)

        entities = build_entities_for_chunk(ner_items)
        if not entities:
            continue

        entities_prompt = format_entities_for_prompt(entities)
        parsed = call_openai_triples(client, text, entities_prompt)

        for t in parsed.get("triples", []):
            try:
                si = int(t["subject_idx"])
                oi = int(t["object_idx"])
                raw_pred = str(t["predicate"])
            except Exception:
                continue

            if si < 0 or si >= len(entities) or oi < 0 or oi >= len(entities):
                continue

            pred = normalize_predicate(raw_pred, pred_map)
            if not pred:
                continue

            subj = entities[si]
            obj = entities[oi]
            conf = float(t.get("confidence")) if t.get("confidence") is not None else None

            triple_id = _hash(f"{doc_id}|{chunk_id}|{subj.text}|{pred}|{obj.text}")

            results.append(TripleOut(
                doc_id=doc_id,
                chunk_id=chunk_id,
                page=page,
                subject=subj.text,
                subject_label=subj.label,
                subject_start=subj.start,
                subject_end=subj.end,
                subject_doc_char_start=subj.doc_char_start,
                subject_doc_char_end=subj.doc_char_end,
                predicate=pred,
                object=obj.text,
                object_label=obj.label,
                object_start=obj.start,
                object_end=obj.end,
                object_doc_char_start=obj.doc_char_start,
                object_doc_char_end=obj.doc_char_end,
                triple_id=triple_id,
                confidence=conf
            ))

    return [asdict(r) for r in dedupe_triples(results)]

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="Path to chunks.json (PDF parser output)")
    ap.add_argument("--ner", required=True, help="Path to ner.json (NER output)")
    ap.add_argument("--out", required=True, help="Path to write triples.json")
    ap.add_argument("--predicate_map", help="Optional JSON map to remap predicates (e.g., {'reduces':'lowers'})")
    args = ap.parse_args()

    pred_map = load_predicate_map(args.predicate_map)
    chunks_by_id = load_chunks(args.chunks)

    with open(args.ner, "r", encoding="utf-8") as f:
        ner_rows = json.load(f)
    ner_by_chunk: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in ner_rows:
        ner_by_chunk[e["chunk_id"]].append(e)
    for k in ner_by_chunk:
        ner_by_chunk[k].sort(key=lambda x: (x.get("start", 0), x.get("end", 0), x.get("label", ""), x.get("text", "")))

    triples = extract_triples(chunks_by_id, ner_by_chunk, pred_map)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(triples, f, indent=2, ensure_ascii=False)

    print(f"✅ Extracted {len(triples)} triples → {args.out}")

if __name__ == "__main__":
    main()
