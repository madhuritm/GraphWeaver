#!/usr/bin/env python3
import os, json, time, argparse, hashlib, re
from typing import Dict, Any, List, Optional, Set, Tuple
from openai import OpenAI
from dataclasses import dataclass, asdict
import traceback
# ----------------------------
# Config
# ----------------------------

MODEL = os.getenv("OPENAI_NER_MODEL", "gpt-5")  # e.g., "gpt-4.1", "gpt-4o"
TEMPERATURE = float(os.getenv("OPENAI_NER_TEMPERATURE", "0"))
MAX_RETRIES = int(os.getenv("OPENAI_NER_MAX_RETRIES", "3"))
RETRY_BACKOFF = float(os.getenv("OPENAI_NER_RETRY_BACKOFF", "2.0"))

LABELS_REGISTRY_DEFAULT = os.getenv("OPENAI_NER_LABELS_REGISTRY", "labels_registry.json")

# Broad, domain-agnostic defaults for PDFs
DEFAULT_BASE_LABELS = [
    # Generic entities
    "PERSON", "ORG", "PRODUCT", "BRAND", "WORK", "EVENT", "TOPIC",
    "PLACE", "LOC", "GPE", "FACILITY",
    # Temporal / numeric
    "DATE", "TIME", "DURATION", "QUANTITY", "NUMBER", "PERCENT", "MONEY",
    # Web / ids
    "EMAIL", "URL", "PHONE_NUMBER", "ACCOUNT_ID", "ORDER_ID",
    # Document structure & citations
    "SECTION", "SUBSECTION", "FIGURE", "TABLE", "CAPTION", "REFERENCE", "CITATION",
    # Business / finance / legal
    "TITLE", "DEPARTMENT", "COST", "REVENUE", "INVOICE", "PURCHASE_ORDER", "CONTRACT", "CLAUSE",
    # Academic / scientific
    "METRIC", "MEASUREMENT", "UNIT", "ALGORITHM", "MODEL", "DATASET",
    # Software / tech
    "FILE_PATH", "CODE_SNIPPET", "API", "LIBRARY", "VERSION",
    # Other common
    "ADDRESS", "CITY", "STATE", "COUNTRY", "ZIP_CODE",
]

SYSTEM_INSTRUCTIONS = (
    "You are an expert information extraction system. "
    "Extract named entities from the given text and return ONLY the JSON per the provided schema. "
    "Entities must be substrings from the input text with exact character spans (0-based, inclusive start, exclusive end). "
    "Avoid overlapping duplicates; merge identical spans and choose the most specific label. "
    "Prefer the provided KNOWN_LABELS. If none is suitable, you MAY propose a NEW label, "
    "but it must be a short, general, reusable UPPER_SNAKE_CASE token (e.g., FILE_PATH, PROJECT_CODE). "
    "Do NOT invent hyper-specific or document-unique labels. "
    "If unsure, omit the item rather than guessing. Confidence is a float in [0,1]."
)

# ----------------------------
# JSON Schema for Structured Outputs
# ----------------------------

def build_response_format_json_schema() -> Dict[str, Any]:
    # Note: label is a free string to allow dynamic/learned labels.
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ner_extraction",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "label": {"type": "string", "minLength": 1},
                                "text": {"type": "string"},
                                "start": {"type": "integer", "minimum": 0},
                                "end": {"type": "integer", "minimum": 0},
                                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                            },
                            "required": ["label", "text", "start", "end"]
                        }
                    }
                },
                "required": ["entities"]
            },
            "strict": True
        }
    }

# ----------------------------
# Data model
# ----------------------------

@dataclass
class Entity:
    doc_id: str
    chunk_id: str
    page: int
    label: str     # normalized label
    text: str
    start: int
    end: int
    doc_char_start: int
    doc_char_end: int
    confidence: Optional[float] = None
    evidence_id: Optional[str] = None

# ----------------------------
# Utils
# ----------------------------

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

_LABEL_CLEAN_RE = re.compile(r"[^A-Z0-9_]+")

def normalize_label(label: str) -> str:
    """Normalize labels to UPPER_SNAKE_CASE without symbols."""
    if not label:
        return ""
    # Convert camelCase or mixed to snake-ish
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", label)
    s = s.replace("-", "_").replace(" ", "_")
    s = s.upper()
    s = _LABEL_CLEAN_RE.sub("", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "UNKNOWN"

def load_labels_registry(path: str) -> List[str]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return sorted({normalize_label(x) for x in data if isinstance(x, str) and x.strip()})
    # If no file, seed with defaults
    return sorted({normalize_label(x) for x in DEFAULT_BASE_LABELS})

def save_labels_registry(path: str, labels: List[str]) -> None:
    labels_sorted = sorted({normalize_label(x) for x in labels if x})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(labels_sorted, f, indent=2, ensure_ascii=False)

def compute_new_labels(before: Set[str], after: Set[str]) -> List[str]:
    return sorted(list(after - before))

# ----------------------------
# OpenAI call
# ----------------------------

def call_openai_ner(client: OpenAI, text: str, known_labels: List[str]) -> Dict[str, Any]:
    """
    Calls the Responses API with Structured Outputs.
    Returns a dict like {"entities": [...]} per schema.
    """
    response_format = build_response_format_json_schema()

    # Add a brief hint with the known labels (helps clustering & reuse)
    labels_hint = ", ".join(known_labels[:200])  # cap display size
    user_prompt = ("Extract named entities from the TEXT below and return ONLY this JSON shape:\n"
                    '{ "entities": [ { "label": str, "text": str, "start": int, "end": int, "confidence": float? } ] }\n'
                    "Rules:\n"
                    "- Spans are 0-based, [start, end) over the original TEXT.\n"
                    "- Use KNOWN_LABELS when possible; otherwise propose a short reusable UPPER_SNAKE_CASE label.\n"
                    "- Do not include any extra keys anywhere.\n\n"
                    "KNOWN_LABELS: {labels_hint}\n\n"
                    "TEXT:\n" + text)
    '''
        "Extract entities from the text below. Use exact spans from the text.\n\n"
        f"KNOWN_LABELS (prefer these if suitable): {labels_hint}\n\n"
        "TEXT:\n" + text
    '''
    
    print("HAS_KEY?", bool(os.getenv("OPENAI_API_KEY")))
    k = os.getenv("OPENAI_API_KEY") or ""
    print("KEY_PREFIX:", k[:7] if k else None)
    print("MODEL:", MODEL)

    

    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp =  client.chat.completions.create(
                model=MODEL,    
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user",   "content": user_prompt}
                ],
                response_format={"type": "json_object"},  # or your schema             
            )
           
             # Chat Completions returns JSON string content at choices[0].message.content
            content = resp.choices[0].message.content if resp.choices else ""
            if not content:
                raise ValueError("Empty content from chat.completions response")


            # Strip code fences if the model wrapped output (defensive)
            txt = content.strip()
            if txt.startswith("```"):
                parts = txt.strip("`").split("\n", 1)
                txt = parts[1] if len(parts) > 1 else parts[0]
            parsed = json.loads(txt)
            if not isinstance(parsed, dict) or "entities" not in parsed or not isinstance(parsed["entities"], list):
                raise ValueError("Parsed response is missing 'entities' field or it is not a list")
            return parsed
           
            

        except Exception:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_BACKOFF * attempt)
# ----------------------------
# Pipeline
# ----------------------------

def extract_ner_for_chunks(
    chunks: List[Dict[str, Any]],
    labels_registry_path: str
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Returns (entities, new_labels_added)
    Persists updated label registry on disk.
    """
    # Load existing labels (or seed defaults)
    existing_labels = load_labels_registry(labels_registry_path)
    existing_set: Set[str] = set(existing_labels)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    results: List[Dict[str, Any]] = []
    discovered_labels: Set[str] = set()

    for ch in chunks:
        text = ch["text"]
        doc_id = ch["doc_id"]
        chunk_id = ch["chunk_id"]
        page = ch.get("page", 0)
        char_start_base = ch.get("char_start", 0)

        ner = call_openai_ner(client, text, existing_labels)
        for item in ner.get("entities", []):
            # Defensive checks5
            try:
                start = int(item["start"])
                end = int(item["end"])
                
            except Exception:
                continue
            if start < 0 or end <= start or end > len(text):
                continue

            raw_label = str(item.get("label") or "").strip()
            norm_label = normalize_label(raw_label)
            if not norm_label:
                continue

            # Track any labels we haven't seen before (for learning)
            if norm_label not in existing_set:
                discovered_labels.add(norm_label)
                existing_set.add(norm_label)
                existing_labels.append(norm_label)  # Keep in-memory list hot-updated

            ent_text = text[start:end]
            conf = float(item.get("confidence")) if item.get("confidence") is not None else None

            e = Entity(
                doc_id=doc_id,
                chunk_id=chunk_id,
                page=page,
                label=norm_label,
                text=ent_text,
                start=start,
                end=end,
                doc_char_start=char_start_base + start,
                doc_char_end=char_start_base + end,
                confidence=conf,
                evidence_id=_hash(f"{doc_id}|{chunk_id}|{start}-{end}|{norm_label}|{ent_text}")
            )
            results.append(asdict(e))

    # Persist the updated registry
    save_labels_registry(labels_registry_path, list(existing_set))
    return results, sorted(discovered_labels)

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="Path to chunks.json from Step 1")
    ap.add_argument("--out", required=True, help="Where to write ner.json")
    ap.add_argument("--labels_registry", default=LABELS_REGISTRY_DEFAULT,
                    help="Path to the persistent labels registry JSON (default: labels_registry.json)")
    ap.add_argument("--new_labels_out", default="new_labels_added.json",
                    help="Where to write the run's newly added labels (delta) for inspection")
    args = ap.parse_args()

    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    ner_entities, new_labels = extract_ner_for_chunks(chunks, args.labels_registry)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(ner_entities, f, indent=2, ensure_ascii=False)

    # Write delta file for review
    with open(args.new_labels_out, "w", encoding="utf-8") as f:
        json.dump(new_labels, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(ner_entities)} entities → {args.out}")
    print(f"Labels registry updated → {args.labels_registry}")
    if new_labels:
        print(f"New labels this run ({len(new_labels)}): {', '.join(new_labels)}")
        print(f"Also saved to → {args.new_labels_out}")
    else:
        print("No new labels discovered this run.")

if __name__ == "__main__":
    main()
