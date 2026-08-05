from pathlib import Path
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


def ingest_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    name = path.name
    suffix = path.suffix.lower()
    chunks = []
    n = 0

    if suffix == ".pdf":
        for page_text, page_num in read_pdf(path):
            for c in chunk_text(page_text):
                chunks.append({
                    "text": c,
                    "file_name": name,
                    "page_num": page_num,
                    "chunk_num": n,
                })
                n += 1
    elif suffix == ".txt":
        for c in chunk_text(read_txt(path)):
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
