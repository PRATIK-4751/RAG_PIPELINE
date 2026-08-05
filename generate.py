import requests
from sentence_transformers import CrossEncoder

URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:31b-cloud"

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None

SYSTEM = """You are a document analysis assistant. Answer questions ONLY based on the provided context. If the context does not contain enough information to answer the question, say "I don't have enough information to answer this question." Do not make up information."""

PROMPT = """Context:
---
{context}
---

Question: {question}

Answer based only on the context above:"""


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def generate(context, question):
    p = PROMPT.format(context=context, question=question)
    r = requests.post(URL, json={
        "model": MODEL,
        "system": SYSTEM,
        "prompt": p,
        "stream": False,
    }, timeout=300)
    if r.status_code != 200:
        raise Exception(f"ollama error: {r.status_code} - {r.text}")
    return r.json()["response"]


def generate_stream(context, question):
    p = PROMPT.format(context=context, question=question)
    r = requests.post(URL, json={
        "model": MODEL,
        "system": SYSTEM,
        "prompt": p,
        "stream": True,
    }, timeout=300, stream=True)
    if r.status_code != 200:
        raise Exception(f"ollama error: {r.status_code} - {r.text}")
    import json
    for line in r.iter_lines():
        if not line:
            continue
        try:
            obj = json.loads(line.decode("utf-8"))
            chunk = obj.get("response", "")
            if chunk:
                yield chunk
            if obj.get("done"):
                break
        except Exception:
            continue


def check():
    try:
        r = requests.get("http://localhost:11434/api/tags")
        if r.status_code != 200:
            return False, []
        names = [m["name"] for m in r.json().get("models", [])]
        # ollama returns either "name" or "name:latest"
        ok = MODEL in names or f"{MODEL}:latest" in names
        return ok, names
    except requests.ConnectionError:
        return False, []


def rewrite_query(question):
    p = f"""Rewrite this question to be clearer and more specific for searching documents. Keep it as a single question. Do not answer it.

Original: {question}

Rewritten:"""
    r = requests.post(URL, json={
        "model": MODEL,
        "system": "You rewrite questions to be clearer. Output only the rewritten question.",
        "prompt": p,
        "stream": False,
    }, timeout=60)
    if r.status_code != 200:
        return question
    text = r.json()["response"].strip()
    # strip quotes if model wrapped it
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text or question


def hyde(question):
    p = f"""Write a short paragraph (2-3 sentences) that would answer this question. This is a hypothetical answer used to help search documents, not a real answer.

Question: {question}

Hypothetical answer:"""
    r = requests.post(URL, json={
        "model": MODEL,
        "system": "You write hypothetical answers to help with document search.",
        "prompt": p,
        "stream": False,
    }, timeout=60)
    if r.status_code != 200:
        return question
    return r.json()["response"].strip() or question


def rerank(query, results, top_n=3):
    if not results:
        return []
    reranker = _get_reranker()
    pairs = [[query, r["text"]] for r in results]
    scores = reranker.predict(pairs).tolist()
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    out = []
    for r, s in ranked[:top_n]:
        new_r = dict(r)
        new_r["rerank_score"] = float(s)
        out.append(new_r)
    return out


def agent_loop(question, retrieve_fn, max_iters=2):
    # simple agent: ask, retrieve, generate, if low confidence refine and retry
    history = []
    q = question
    for i in range(max_iters):
        results = retrieve_fn(q)
        if not results:
            break
        ctx = "\n\n".join(r["text"] for r in results)
        ans = generate(ctx, q)
        history.append({"iter": i + 1, "query": q, "answer": ans, "n_chunks": len(results)})
        # cheap stop: if iter 0 produced a decent answer, stop
        if i == 0 and "I don't have enough information" not in ans:
            break
        # refine query
        q = rewrite_query(q)
    return ans, history
