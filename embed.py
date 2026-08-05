from sentence_transformers import SentenceTransformer
import chromadb

MODEL = "all-MiniLM-L6-v2"
DB_PATH = "chroma_db"
COLLECTION = "documents"


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer(MODEL)
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks):
        if not chunks:
            return 0

        ids = []
        texts = []
        metas = []

        for c in chunks:
            ids.append(f"{c['file_name']}_{c['chunk_num']}")
            texts.append(c["text"])
            page = c.get("page_num")
            # chromadb doesn't like None in metadata
            metas.append({
                "source": c["file_name"],
                "page": str(page) if page is not None else "",
                "chunk_num": c.get("chunk_num", 0),
            })

        embeddings = self.model.encode(texts).tolist()
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metas,
        )
        return len(texts)

    def search(self, query, k=5):
        emb = self.model.encode([query]).tolist()
        res = self.collection.query(
            query_embeddings=emb,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        out = []
        for i in range(len(res["ids"][0])):
            out.append({
                "text": res["documents"][0][i],
                "source": res["metadatas"][0][i]["source"],
                "page": res["metadatas"][0][i]["page"],
                # 1 - distance gives similarity for cosine
                "score": 1 - res["distances"][0][i],
            })
        return out

    def count(self):
        return self.collection.count()

    def clear(self):
        self.client.delete_collection(COLLECTION)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
