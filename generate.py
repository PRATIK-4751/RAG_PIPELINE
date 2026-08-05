import requests

URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:31b-cloud"

SYSTEM = """You are a document analysis assistant. Answer questions ONLY based on the provided context. If the context does not contain enough information to answer the question, say "I don't have enough information to answer this question." Do not make up information."""

PROMPT = """Context:
---
{context}
---

Question: {question}

Answer based only on the context above:"""


def generate(context, question):
    p = PROMPT.format(context=context, question=question)
    r = requests.post(URL, json={
        "model": MODEL,
        "system": SYSTEM,
        "prompt": p,
        "stream": False,
    }, timeout=120)
    if r.status_code != 200:
        raise Exception(f"ollama error: {r.status_code} - {r.text}")
    return r.json()["response"]


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
