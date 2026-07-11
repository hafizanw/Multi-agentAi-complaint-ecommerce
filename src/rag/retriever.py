import chromadb
from src.rag.embedder import LocalEmbedder
from src.config import VECTORSTORE_PATHS, COLLECTION_NAMES


class Retriever:
    """
    Retriever generik per domain. Setiap instance HANYA terhubung ke satu
    PersistentClient/collection sesuai domain-nya -> ini yang mewujudkan
    "silo keamanan data divisi" di level kode, bukan cuma di level desain.
    """

    def __init__(self, domain: str):
        if domain not in VECTORSTORE_PATHS:
            raise ValueError(f"Domain tidak dikenal: {domain}")
        self.domain = domain
        self.embedder = LocalEmbedder()
        self.client = chromadb.PersistentClient(path=VECTORSTORE_PATHS[domain])
        self.collection = self.client.get_collection(COLLECTION_NAMES[domain])

    def query(self, question: str, top_k: int = 3):
        query_emb = self.embedder.embed([question])[0]
        results = self.collection.query(query_embeddings=[query_emb], n_results=top_k)
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        return docs, metas
