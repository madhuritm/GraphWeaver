#!/usr/bin/env python3
import os, json, hashlib, argparse, re
from typing import Dict, Any
from neo4j import GraphDatabase

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def node_id(doc_id: str, label: str, text: str) -> str:
    return _sha1(f"{doc_id}|{label}|{_norm_space(text)}")

def to_rel_type(pred: str) -> str:
    # Neo4j rel types must be ASCII uppercase + underscores; keep it generic
    s = pred.strip().upper()
    s = re.sub(r"[^A-Z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "RELATED_TO"

def run_ingest(uri: str, user: str, password: str, triples_path: str, maingraphnode: str, batch_size: int = 500):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver, driver.session() as sess:
        with open(triples_path, "r", encoding="utf-8") as f:
            triples = json.load(f)

        # Batch for speed
        for i in range(0, len(triples), batch_size):
            chunk = triples[i:i+batch_size]
            tx = sess.begin_transaction()
            for t in chunk:
                doc_id   = t["doc_id"]
                chunk_id = t["chunk_id"]
                page     = t.get("page")
                sent     = t.get("sentence","")
                s_lab    = t["subject_label"]
                s_txt    = t["subject"]
                o_lab    = t["object_label"]
                o_txt    = t["object"]
                pred_raw = t["predicate"]
                rel_type = to_rel_type(pred_raw)
                triple_id= t.get("triple_id")

                s_id = node_id(doc_id, s_lab, s_txt)
                o_id = node_id(doc_id, o_lab, o_txt)

                # MERGE subject
                tx.run(
                    f"""
                    MERGE (s:{maingraphnode}:{s_lab} {{id:$sid}})
                    ON CREATE SET s.text=$stxt,s.name=$stxt,s.label=$slab, s.createdAt=timestamp()
                    ON MATCH  SET s.text=$stxt, s.label=$slab
                    """,
                    sid=s_id, stxt=s_txt, slab=s_lab
                )

                # MERGE object
                tx.run(
                    f"""
                    MERGE (o:{maingraphnode}:{o_lab} {{id:$oid}})
                    ON CREATE SET o.text=$otxt, o.name=$otxt, o.label=$olab, o.createdAt=timestamp()
                    ON MATCH  SET o.text=$otxt, o.label=$olab
                    """,
                    oid=o_id, otxt=o_txt, olab=o_lab
                )

                # MERGE relationship with evidence
                # Note: cannot have a universal relationship uniqueness constraint across all types,
                # so we rely on pipeline dedupe + triple_id uniqueness in data.
                tx.run(
                    f"""
                    MATCH (s:{maingraphnode} {{id:$sid}}), (o:{maingraphnode} {{id:$oid}})
                    MERGE (s)-[r:{rel_type} {{triple_id:$tid}}]->(o)
                    ON CREATE SET
                      r.doc_id=$doc_id,
                      r.page=$page,
                      r.chunk_id=$chunk_id,
                      r.sentence=$sentence,
                      r.subject_span=$sspan,
                      r.object_span=$ospan,
                      r.confidence=$confidence,
                      r.createdAt=timestamp()
                    ON MATCH SET
                      r.doc_id=$doc_id,
                      r.page=$page,
                      r.chunk_id=$chunk_id,
                      r.sentence=$sentence,
                      r.subject_span=$sspan,
                      r.object_span=$ospan,
                      r.confidence=$confidence
                    """,
                    sid=s_id,
                    oid=o_id,
                    tid=triple_id,
                    doc_id=doc_id,
                    page=page,
                    chunk_id=chunk_id,
                    sentence=sent,
                    sspan=t.get("subject_span_in_sentence"),
                    ospan=t.get("object_span_in_sentence"),
                    confidence=t.get("confidence"),
                )
            tx.commit()
    print("Neo4j ingest complete.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True, help="neo4j+s://... (Aura) or bolt+s://...")
    ap.add_argument("--user", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--triples", required=True, help="Path to triples.json")
    ap.add_argument("--maingraphnode", required=True, help="Name of the main Graph node in neo4j")
    ap.add_argument("--batch", type=int, default=500)
    args = ap.parse_args()
    run_ingest(args.uri, args.user, args.password, args.triples, args.maingraphnode, args.batch)

if __name__ == "__main__":
    main()
