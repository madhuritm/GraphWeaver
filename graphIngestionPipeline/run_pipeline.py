#!/usr/bin/env python3
import os, subprocess, argparse, pathlib, sys

def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--doc-id", required=True)    
    ap.add_argument("--work", default="../out")
    ap.add_argument("--main_label", required=True)
    ap.add_argument("--max-tokens", type=int, default=1000, help="Target tokens per chunk")
    args = ap.parse_args()

    workdir = pathlib.Path(args.work) / args.doc_id
    workdir.mkdir(parents=True, exist_ok=True)
    chunks = workdir/"chunks.json"
    ner = workdir/"ner.json"
    labels_registry = workdir/"labels_registry.json"
    new_labels = workdir/"new_labels_added.json"
    triples=workdir/"triples.json"

    run(["python", "scripts/pdf_parser.py", "--pdf", args.pdf, "--doc-id", str(args.doc_id), "--out", str(chunks), "--max-tokens", str(args.max_tokens)])
    run(["python", "scripts/NER.py", "--chunks", str(chunks), "--out", str(ner), "--labels_registry", str(labels_registry), "--new_labels_out", str(new_labels)])
    run(["python", "scripts/triples_gen.py", "--chunks", str(chunks), "--ner", str(ner), "--out", str(triples)])
    run([
        "python", "scripts/neo4j_ingestor.py",
        "--uri", os.getenv("NEO4J_URI"),
        "--user", os.getenv("NEO4J_USER"),
        "--password", os.getenv("NEO4J_PASS"),
        "--triples", str(triples),
        "--maingraphnode", args.main_label,
    ])
    
    
    print("done:", workdir)

if __name__ == "__main__":
    sys.exit(main())
    
