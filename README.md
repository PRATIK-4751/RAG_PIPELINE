# local rag pipeline

ask questions to your pdfs and txt files. runs 100% on your machine, no api keys.

## what it does

you give it a pdf or text file, ask a question, and it answers using only what's in the file. it embeds the text with sentence-transformers, stores vectors in chromadb, and uses ollama to generate the final answer.

## setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama pull gemma4:31b-cloud
```

## usage

```bash
python main.py ingest --file data/sample.txt
python main.py query "what is RAG?"
python main.py query "what are benefits?" --show-chunks
python main.py query "what are benefits?" --no-rewrite
python main.py query "what are benefits?" --hyde
python main.py compare "how do the documents describe x?"
python main.py ask "what is RAG?"
python main.py status
python main.py clear
python main.py export "what is RAG?" --output out.json
```

query uses hybrid search (semantic + keyword), a query rewrite, and a cross-encoder rerank by default. pass `--no-rewrite` to skip the rewrite, `--hyde` to search with a hypothetical answer instead, or `--no-rerank` to skip the reranker. compare mode asks the model to find agreements and disagreements across the documents and writes one answer. ask mode runs a tiny agent loop: it answers, checks for a confident answer, and refines the query once if the first answer is weak.

there is also a chat ui:

```bash
python app.py
```

it streams answers, keeps a short memory of the conversation, and has an advanced toggle to turn the rewrite and rerank on or off. the memory resets when you press clear.

## eval

```bash
python evaluate.py
python evaluate.py --pipeline --limit 3
python evaluate.py --pipeline --output eval/report.json
```

`evaluate.py` reads the questions in `eval/questions.json`, runs each one, and scores retrieval hit, keyword match, and answer quality with a judge. `--pipeline` turns on the same rewrite + rerank the cli uses, `--limit` stops early, and `--output` writes a report.

## files

- `ingest.py` - reads pdf/txt, splits into chunks
- `embed.py` - vector store wrapper around chromadb
- `retrieve.py` - search, hybrid search, confidence and citation helpers
- `generate.py` - ollama api calls, streaming, reranking
- `synthesis.py` - agreement/contradiction checks for compare mode
- `memory.py` - short conversation memory for the ui
- `app.py` - gradio chat ui
- `evaluate.py` - runs the questions in eval/questions.json and scores them
- `main.py` - cli

## notes

- only text-based pdfs work, scanned ones need ocr first
- chunk size is 500 chars with 100 char overlap, change in `ingest.py`
- change model in `generate.py` (`MODEL = "..."`)
- needs ~4-6gb vram for 9b models
- reranking downloads a small cross-encoder model on first use
