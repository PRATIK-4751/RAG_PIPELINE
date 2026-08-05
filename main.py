import argparse
import json
from pathlib import Path

from ingest import ingest_file
from embed import VectorStore
from retrieve import search, format_context, get_sources, get_confidence
from generate import generate, check


def do_ingest(args):
    p = Path(args.file)
    if not p.exists():
        print(f"file not found: {p}")
        return
    store = VectorStore()
    chunks = ingest_file(p)
    n = store.add(chunks)
    print(f"ingested {n} chunks from {p.name}")
    print(f"total: {store.count()}")


def do_query(args):
    ok, _ = check()
    if not ok:
        print("ollama not running, start it first")
        return

    results = search(args.question, k=args.top_k)
    if not results:
        print("no chunks found")
        return

    ctx = format_context(results)
    print(f"confidence: {get_confidence(results)}")
    print(f"sources: {', '.join(get_sources(results))}\n")

    ans = generate(ctx, args.question)
    print(f"answer: {ans}\n")

    if args.show_chunks:
        print("chunks used:")
        for i, r in enumerate(results):
            print(f"  {i+1}. [{r['score']:.3f}] {r['source']}: {r['text'][:80]}...")


def do_status(args):
    store = VectorStore()
    print(f"vector store: {store.count()} chunks")
    ok, models = check()
    if ok:
        print(f"ollama: connected (models: {', '.join(models)})")
    else:
        print("ollama: not running")


def do_clear(args):
    VectorStore().clear()
    print("cleared")


def do_export(args):
    results = search(args.question, k=args.top_k)
    if not results:
        print("no chunks found")
        return
    ctx = format_context(results)
    ans = generate(ctx, args.question)
    data = {
        "question": args.question,
        "answer": ans,
        "confidence": get_confidence(results),
        "sources": get_sources(results),
        "chunks": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"exported to {args.output}")


def main():
    p = argparse.ArgumentParser(description="local rag pipeline")
    sub = p.add_subparsers(dest="cmd")

    pi = sub.add_parser("ingest")
    pi.add_argument("--file", "-f", required=True)
    pi.set_defaults(func=do_ingest)

    pq = sub.add_parser("query")
    pq.add_argument("question")
    pq.add_argument("--top-k", "-k", type=int, default=5)
    pq.add_argument("--show-chunks", "-s", action="store_true")
    pq.set_defaults(func=do_query)

    ps = sub.add_parser("status")
    ps.set_defaults(func=do_status)

    pc = sub.add_parser("clear")
    pc.set_defaults(func=do_clear)

    pe = sub.add_parser("export")
    pe.add_argument("question")
    pe.add_argument("--output", "-o", default="export.json")
    pe.add_argument("--top-k", "-k", type=int, default=5)
    pe.set_defaults(func=do_export)

    args = p.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
