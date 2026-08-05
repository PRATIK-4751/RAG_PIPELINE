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
python main.py status
python main.py clear
python main.py export "what is RAG?" --output out.json
```

## files

- `ingest.py` - reads pdf/txt, splits into chunks
- `embed.py` - vector store wrapper around chromadb
- `retrieve.py` - search and format helpers
- `generate.py` - ollama api calls
- `evaluate.py` - runs the questions in eval/questions.json and scores them
- `main.py` - cli

## notes

- only text-based pdfs work, scanned ones need ocr first
- chunk size is 500 chars with 100 char overlap, change in `ingest.py`
- change model in `generate.py` (`MODEL = "..."`)
- needs ~4-6gb vram for 9b models
- chromadb 0.5.9 is pinned because newer versions break on windows
