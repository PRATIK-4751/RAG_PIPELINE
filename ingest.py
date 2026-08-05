from pathlib import Path
import re
import pdfplumber


def read_pdf(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text and text.strip():
                pages.append((text, i))
    return pages


def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text, size=500, overlap=100):
    out = []
    i = 0
    while i < len(text):
        piece = text[i:i + size].strip()
        if piece:
            out.append(piece)
        i += size - overlap
    return out


def split_paragraphs(text):
    # split on double newlines or single newlines followed by sentence end
    paras = re.split(r"\n\s*\n", text)
    out = []
    for p in paras:
        p = p.strip()
        if p:
            out.append(p)
    return out


def split_sentences(text):
    # simple sentence splitter
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


def recursive_chunk(text, size=500, overlap=100):
    # try paragraphs first, then sentences, then chars
    pieces = split_paragraphs(text)
    if not pieces:
        return chunk_text(text, size, overlap)

    out = []
    current = ""
    for p in pieces:
        if len(current) + len(p) + 1 <= size:
            current = (current + "\n" + p).strip()
        else:
            if current:
                out.append(current)
            if len(p) > size:
                # paragraph too big, fall back to sentences
                sents = split_sentences(p)
                cur = ""
                for s in sents:
                    if len(cur) + len(s) + 1 <= size:
                        cur = (cur + " " + s).strip()
                    else:
                        if cur:
                            out.append(cur)
                        if len(s) > size:
                            # sentence too big, char split
                            out.extend(chunk_text(s, size, overlap))
                            cur = ""
                        else:
                            cur = s
                if cur:
                    out.append(cur)
                current = ""
            else:
                current = p
    if current:
        out.append(current)

    # add overlap
    if overlap > 0 and len(out) > 1:
        overlapped = [out[0]]
        for i in range(1, len(out)):
            tail = overlapped[-1][-overlap:] if len(overlapped[-1]) > overlap else overlapped[-1]
            overlapped.append((tail + " " + out[i]).strip())
        out = overlapped

    return out


def ingest_file(path, mode="recursive"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    name = path.name
    suffix = path.suffix.lower()
    chunks = []
    n = 0

    chunk_fn = recursive_chunk if mode == "recursive" else chunk_text

    if suffix == ".pdf":
        for page_text, page_num in read_pdf(path):
            for c in chunk_fn(page_text):
                chunks.append({
                    "text": c,
                    "file_name": name,
                    "page_num": page_num,
                    "chunk_num": n,
                })
                n += 1
    elif suffix == ".txt":
        for c in chunk_fn(read_txt(path)):
            chunks.append({
                "text": c,
                "file_name": name,
                "page_num": None,
                "chunk_num": n,
            })
            n += 1
    else:
        raise ValueError(f"unsupported file type: {suffix}")

    return chunks
