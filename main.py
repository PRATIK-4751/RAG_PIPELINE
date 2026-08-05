import argparse
import json
from pathlib import Path

from ingest import ingest_file
from embed import VectorStore
from retrieve import format_context, get_sources, calibrate_confidence, verify_citations
from generate import generate, check, agent_loop
from pipeline import retrieve_top
from synthesis import detect_contradictions, synthesize


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

    q, results = retrieve_top(args.question, k=args.top_k, rewrite=not args.no_rewrite, use_hyde=args.hyde, rerank=not args.no_rerank)

    if not results:
        print("no chunks found")
        return

    ctx = format_context(results)
    ans = generate(ctx, args.question)

    conf = calibrate_confidence(results, ans, ctx)
    cites = verify_citations(ans, results)

    if q != args.question:
        print(f"search query: {q}")
    print(f"confidence: {conf['label']} ({conf['score']})")
    print(f"sources: {', '.join(get_sources(results))}\n")

    print(f"answer: {ans}\n")

    if cites["total_cited"]:
        print(f"citations: verified {len(cites['verified'])}, unverified {len(cites['unverified'])}")

    if args.show_chunks:
        print("chunks used:")
        for i, r in enumerate(results):
            rs = f" rerank={r['rerank_score']:.3f}" if r.get("rerank_score") else ""
            print(f"  {i+1}. [{r['score']:.3f}{rs}] {r['source']}: {r['text'][:80]}...")


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


def do_compare(args):
    ok, _ = check()
    if not ok:
        print("ollama not running, start it first")
        return

    q, results = retrieve_top(args.question, k=args.top_k, rewrite=not args.no_rewrite, use_hyde=args.hyde, rerank=not args.no_rerank)
    if not results:
        print("no chunks found")
        return

    claims = [{"text": r["text"], "source": r["source"], "page": r.get("page")} for r in results]
    agreements = detect_contradictions(claims, args.question)
    ans = synthesize(args.question, claims, agreements)

    if q != args.question:
        print(f"search query: {q}")
    print(f"answer: {ans}\n")
    print(f"sources: {', '.join(get_sources(results))}\n")

    if agreements:
        print("agreements found:")
        for a in agreements:
            level = a.get("level", "unknown")
            srcs = ", ".join(a.get("sources", []))
            claim = a.get("claim", "")[:80]
            print(f"  [{level}] {srcs}: {claim}")
    else:
        print("no agreements parsed")


def do_export(args):
    q, results = retrieve_top(args.question, k=args.top_k, rewrite=not args.no_rewrite, use_hyde=args.hyde, rerank=not args.no_rerank)
    if not results:
        print("no chunks found")
        return
    ctx = format_context(results)
    ans = generate(ctx, args.question)
    conf = calibrate_confidence(results, ans, ctx)
    cites = verify_citations(ans, results)
    data = {
        "question": args.question,
        "search_query": q if q != args.question else None,
        "answer": ans,
        "confidence": conf,
        "sources": get_sources(results),
        "citations": cites,
        "chunks": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"exported to {args.output}")


def do_ask(args):
    ok, _ = check()
    if not ok:
        print("ollama not running, start it first")
        return

    q = args.followup if args.followup else args.question
    retrieve_fn = lambda q: retrieve_top(q, k=args.top_k)[1]
    ans, history = agent_loop(q, retrieve_fn)

    print(f"answer: {ans}")
    followups = [h["query"] for h in history[1:]]
    if followups:
        print(f"follow ups: {', '.join(followups)}")


def main():
    p = argparse.ArgumentParser(description="local rag pipeline")
    sub = p.add_subparsers(dest="cmd")

    pi = sub.add_parser("ingest")
    pi.add_argument("--file", "-f", required=True)
    pi.set_defaults(func=do_ingest)

    pq = sub.add_parser("query")
    pq.add_argument("question")
    pq.add_argument("--top-k", "-k", type=int, default=5)
    pq.add_argument("--no-rerank", action="store_true")
    pq.add_argument("--no-rewrite", action="store_true")
    pq.add_argument("--hyde", action="store_true")
    pq.add_argument("--show-chunks", "-s", action="store_true")
    pq.set_defaults(func=do_query)

    ps = sub.add_parser("status")
    ps.set_defaults(func=do_status)

    pc = sub.add_parser("clear")
    pc.set_defaults(func=do_clear)

    pcm = sub.add_parser("compare")
    pcm.add_argument("question")
    pcm.add_argument("--top-k", "-k", type=int, default=5)
    pcm.add_argument("--no-rerank", action="store_true")
    pcm.add_argument("--no-rewrite", action="store_true")
    pcm.add_argument("--hyde", action="store_true")
    pcm.set_defaults(func=do_compare)

    pe = sub.add_parser("export")
    pe.add_argument("question")
    pe.add_argument("--output", "-o", default="export.json")
    pe.add_argument("--top-k", "-k", type=int, default=5)
    pe.add_argument("--no-rerank", action="store_true")
    pe.add_argument("--no-rewrite", action="store_true")
    pe.add_argument("--hyde", action="store_true")
    pe.set_defaults(func=do_export)

    pa = sub.add_parser("ask")
    pa.add_argument("question")
    pa.add_argument("--followup", "-f")
    pa.add_argument("--top-k", "-k", type=int, default=5)
    pa.set_defaults(func=do_ask)

    args = p.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
