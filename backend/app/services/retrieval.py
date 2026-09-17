import chromadb
from sentence_transformers import SentenceTransformer
from app.core.config import settings

_client = None
_collection = None
_embedder = None


def load_vector_store():
    """Load the persisted Chroma vector store and embedding model once at startup."""
    global _client, _collection, _embedder
    _client = chromadb.PersistentClient(path=settings.vector_store_path)
    _collection = _client.get_or_create_collection(settings.collection_name)
    _embedder = SentenceTransformer(settings.embedding_model)
    return _collection


def get_collection():
    if _collection is None:
        raise RuntimeError("Vector store not loaded. Call load_vector_store() at startup.")
    return _collection


def retrieve(question: str, k: int = 3, appliance_hint: str | None = None):
    if _embedder is None or _collection is None:
        raise RuntimeError("Vector store not loaded. Call load_vector_store() at startup.")

    query_embedding = _embedder.encode([question])[0].tolist()
    where = {"source": appliance_hint} if appliance_hint else None

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
    )

    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    for doc, meta in zip(docs, metas):
        hits.append({"text": doc, "source": meta.get("source", "unknown")})
    return hits
