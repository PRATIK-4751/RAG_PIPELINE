import json
import re

import requests

from retrieve import hybrid_search, format_context, get_sources, calibrate_confidence, verify_citations
from generate import generate, check

URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:31b-cloud"


def keyword_match(answer, keywords):
    if not keywords:
        return 0
    a = answer.lower()
    return sum(1 for k in keywords if k.lower() in a) / len(keywords)


def llm_judge(question, answer, context):
    p = (
        "Score the answer on two scales, each from 0 to 1.\n"
        "faithfulness: does the answer stay true to the context\n"
        "relevance: does the answer answer the question\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context[:3000]}\n\n"
        f"Answer:\n{answer}\n\n"
        'Reply with JSON only: {"faithfulness": 0.0, "relevance": 0.0}'
    )
    try:
        r = requests.post(URL, json={
            "model": MODEL,
            "system": "You score answers. Output JSON only.",
            "prompt": p,
            "stream": False,
        }, timeout=120)
        if r.status_code != 200:
            return {"faithfulness": None, "relevance": None}
        text = r.json()["response"].strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {"faithfulness": None, "relevance": None}
        data = json.loads(m.group(0))
        return {
            "faithfulness": float(data.get("faithfulness", 0.0)),
            "relevance": float(data.get("relevance", 0.0)),
        }
    except Exception:
        return {"faithfulness": None, "relevance": None}


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

        retrieved = hybrid_search(q["question"], k)
        ctx = format_context(retrieved)
        ans = generate(ctx, q["question"])
        judge = llm_judge(q["question"], ans, ctx)
        conf = calibrate_confidence(retrieved, ans, ctx)
        cites = verify_citations(ans, retrieved)

        results.append({
            "question": q["question"],
            "expected_keywords": q["expected_keywords"],
            "answer": ans,
            "confidence": conf["label"],
            "confidence_score": conf["score"],
            "keyword_overlap": keyword_match(ans, q["expected_keywords"]),
            "llm_judge": judge,
            "citations": cites,
            "sources": get_sources(retrieved),
        })
    return results


def show(results):
    if not results:
        return

    print("\n" + "=" * 60)
    print("EVAL RESULTS")
    print("=" * 60)

    total = 0
    passed = 0
    jf = [j["faithfulness"] for r in results for j in [r["llm_judge"]] if j["faithfulness"] is not None]
    jr = [j["relevance"] for r in results for j in [r["llm_judge"]] if j["relevance"] is not None]

    for i, r in enumerate(results):
        total += r["keyword_overlap"]
        ok = r["keyword_overlap"] >= 0.5
        if ok:
            passed += 1

        j = r["llm_judge"]
        jstr = "n/a"
        if j["faithfulness"] is not None:
            jstr = f"f={j['faithfulness']:.2f} r={j['relevance']:.2f}"

        print(f"\n{i+1}. [{'PASS' if ok else 'FAIL'}] {r['question']}")
        print(f"   conf: {r['confidence']} ({r['confidence_score']}) | overlap: {r['keyword_overlap']:.0%} | judge: {jstr}")
        ans = r['answer'][:120].replace('\n', ' ')
        print(f"   answer: {ans}...")

    print(f"total: {len(results)}")
    print(f"avg overlap: {total/len(results):.0%}")
    print(f"passed: {passed}/{len(results)}")
    if jf:
        print(f"avg faithfulness: {sum(jf)/len(jf):.2f}")
    if jr:
        print(f"avg relevance: {sum(jr)/len(jr):.2f}")
    print("=" * 60)


def save(results, path="eval/report.json"):
    if not results:
        return
    avg = sum(r["keyword_overlap"] for r in results) / len(results)
    passed = sum(1 for r in results if r["keyword_overlap"] >= 0.5)
    jf = [r["llm_judge"]["faithfulness"] for r in results if r["llm_judge"]["faithfulness"] is not None]
    jr = [r["llm_judge"]["relevance"] for r in results if r["llm_judge"]["relevance"] is not None]
    report = {
        "total_questions": len(results),
        "avg_keyword_overlap": avg,
        "pass_rate": passed / len(results),
        "avg_faithfulness": sum(jf) / len(jf) if jf else None,
        "avg_relevance": sum(jr) / len(jr) if jr else None,
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
