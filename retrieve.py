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
