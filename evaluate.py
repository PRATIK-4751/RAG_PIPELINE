import json
from retrieve import search, format_context, get_sources, get_confidence
from generate import generate, check


def keyword_match(answer, keywords):
    if not keywords:
        return 0
    a = answer.lower()
    return sum(1 for k in keywords if k.lower() in a) / len(keywords)


def run(path="eval/questions.json", k=5):
    with open(path, "r", encoding="utf-8") as f:
        qs = json.load(f)

    ok, _ = check()
    if not ok:
        print("ollama not running, start it first")
        return None

    results = []
    for i, q in enumerate(qs):
        print(f"  [{i+1}/{len(qs)}] {q['question'][:50]}...")

        retrieved = search(q["question"], k)
        ctx = format_context(retrieved)
        ans = generate(ctx, q["question"])

        results.append({
            "question": q["question"],
            "expected_keywords": q["expected_keywords"],
            "answer": ans,
            "confidence": get_confidence(retrieved),
            "keyword_overlap": keyword_match(ans, q["expected_keywords"]),
            "sources": get_sources(retrieved),
        })
    return results


def show(results):
    if not results:
        return

    print("\n" + "=" * 50)
    print("EVAL RESULTS")
    print("=" * 50)

    total = 0
    passed = 0

    for i, r in enumerate(results):
        total += r["keyword_overlap"]
        ok = r["keyword_overlap"] >= 0.5
        if ok:
            passed += 1

        print(f"\n{i+1}. [{'PASS' if ok else 'FAIL'}] {r['question']}")
        print(f"   conf: {r['confidence']} | overlap: {r['keyword_overlap']:.0%}")
        ans = r['answer'][:120].replace('\n', ' ')
        print(f"   answer: {ans}...")

    print("\n" + "=" * 50)
    print(f"total: {len(results)}")
    print(f"avg overlap: {total/len(results):.0%}")
    print(f"passed: {passed}/{len(results)}")
    print("=" * 50)


def save(results, path="eval/report.json"):
    if not results:
        return
    avg = sum(r["keyword_overlap"] for r in results) / len(results)
    passed = sum(1 for r in results if r["keyword_overlap"] >= 0.5)
    report = {
        "total_questions": len(results),
        "avg_keyword_overlap": avg,
        "pass_rate": passed / len(results),
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"saved to {path}")


if __name__ == "__main__":
    r = run()
    if r:
        show(r)
        save(r)
