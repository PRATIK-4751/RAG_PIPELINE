from embed import VectorStore


def search(query, k=5):
    return VectorStore().search(query, k)


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
