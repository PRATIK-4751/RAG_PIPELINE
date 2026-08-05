import requests

URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:31b-cloud"

SYSTEM = """You compare claims from multiple documents. For each claim, decide if sources agree, disagree, partially agree, or are unrelated. Output JSON only."""


def detect_contradictions(claims, question):
    if not claims:
        return []

    # group by source
    by_source = {}
    for c in claims:
        by_source.setdefault(c.get("source", "unknown"), []).append(c["text"])

    p = f"""Question: "{question}"

Claims grouped by source:
"""
    for src, texts in by_source.items():
        p += f"\n--- {src} ---\n"
        for i, t in enumerate(texts[:3]):
            p += f"[{i+1}] {t[:300]}\n"

    p += """
For each distinct claim, say if sources agree, contradict, partially agree, or are unrelated.
Return a JSON array like:
[{"claim": "...", "level": "agreement|contradiction|partial|unrelated", "sources": ["doc1", "doc2"], "explanation": "..."}]
Output ONLY the JSON array, no other text.
"""

    r = requests.post(URL, json={
        "model": MODEL,
        "system": SYSTEM,
        "prompt": p,
        "stream": False,
    }, timeout=120)
    if r.status_code != 200:
        return []

    text = r.json()["response"].strip()
    # try to find json in the response
    import json
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start:end+1])
    except Exception:
        return []


def synthesize(question, claims, agreements):
    if not claims:
        return "no chunks to synthesize"

    ctx_parts = []
    for i, c in enumerate(claims):
        src = f"Source: {c.get('source', 'unknown')}"
        if c.get("page"):
            src += f", Page: {c['page']}"
        ctx_parts.append(f"[{i+1}] {src}\n{c['text']}")
    ctx = "\n\n".join(ctx_parts)

    note = ""
    if agreements:
        note = "\n\nKnown agreement info:\n"
        for a in agreements[:5]:
            note += f"- {a.get('level', '')}: {a.get('claim', '')[:150]}\n"

    p = f"""Context from multiple documents:
{ctx}
{note}

Question: {question}

Write a single answer that:
- synthesizes info across the sources
- cites with [1], [2] etc.
- if sources disagree, mention both sides
- if info is missing, say so
- do not make things up

Answer:"""

    r = requests.post(URL, json={
        "model": MODEL,
        "system": "You synthesize info from multiple documents. Be honest about disagreements.",
        "prompt": p,
        "stream": False,
    }, timeout=300)
    if r.status_code != 200:
        return f"ollama error: {r.status_code}"
    return r.json()["response"]
