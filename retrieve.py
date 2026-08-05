from embed import VectorStore


def search(query, k=5):
    return VectorStore().search(query, k)


def bm25_search(query, k=5):
    return VectorStore().bm25_search(query, k)


def hybrid_search(query, k=5, alpha=0.7):
    store = VectorStore()
    vec = store.search(query, k=k * 2)
    bm = store.bm25_search(query, k=k * 2)

    combined = {}

    for r in vec:
        key = r["text"]
        combined[key] = {
            "text": r["text"],
            "source": r["source"],
            "page": r["page"],
            "vec": r["score"],
            "bm": 0.0,
        }

    if bm:
        max_bm = max(r["score"] for r in bm)
        for r in bm:
            key = r["text"]
            norm = r["score"] / max_bm if max_bm > 0 else 0
            if key in combined:
                combined[key]["bm"] = norm
            else:
                combined[key] = {
                    "text": r["text"],
                    "source": r["source"],
                    "page": r["page"],
                    "vec": 0.0,
                    "bm": norm,
                }

    out = []
    for r in combined.values():
        r["score"] = alpha * r["vec"] + (1 - alpha) * r["bm"]
        out.append(r)

    out.sort(key=lambda r: r["score"], reverse=True)
    return out[:k]


def format_context(results):
    parts = []
    for i, r in enumerate(results):
        src = f"source: {r['source']}"
        if r.get("page"):
            src += f", page: {r['page']}"
        parts.append(f"[Chunk {i+1} - {src}]\n{r['text']}")
    return "\n\n".join(parts)


def get_sources(results):
    seen = []
    for r in results:
        s = r["source"]
        if r.get("page"):
            s += f" (page {r['page']})"
        if s not in seen:
            seen.append(s)
    return seen


def get_confidence(results):
    if not results:
        return "none"
    avg = sum(r["score"] for r in results) / len(results)
    if avg > 0.6:
        return "high"
    if avg > 0.4:
        return "medium"
    return "low"


def calibrate_confidence(results, answer, context):
    # combine retrieval score and a cheap signal from the answer
    if not results:
        return {"label": "none", "score": 0.0, "signals": {}}

    sim = sum(r["score"] for r in results) / len(results)
    n_unique = len(set(r["source"] for r in results))
    diversity = min(n_unique / 3.0, 1.0)

    # check if the answer says it doesn't have info
    refused = "I don't have enough information" in answer or "I do not have enough information" in answer
    coverage = 0.3 if refused else 0.8

    score = 0.5 * sim + 0.2 * diversity + 0.3 * coverage
    score = max(0.0, min(1.0, score))

    if score > 0.6:
        label = "high"
    elif score > 0.4:
        label = "medium"
    else:
        label = "low"

    return {
        "label": label,
        "score": round(score, 3),
        "signals": {
            "similarity": round(sim, 3),
            "diversity": round(diversity, 3),
            "coverage": round(coverage, 3),
        },
    }


def verify_citations(answer, results):
    # find [1], [2], etc in the answer and check they map to retrieved chunks
    import re
    cited = re.findall(r"\[(\d+)\]", answer)
    n_chunks = len(results)
    verified = []
    unverified = []
    for c in cited:
        try:
            i = int(c) - 1
            if 0 <= i < n_chunks:
                verified.append(int(c))
            else:
                unverified.append(int(c))
        except ValueError:
            unverified.append(c)
    return {
        "verified": sorted(set(verified)),
        "unverified": sorted(set(unverified)),
        "total_cited": len(set(cited)),
    }
