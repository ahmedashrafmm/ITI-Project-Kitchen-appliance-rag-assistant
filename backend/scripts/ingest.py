"""Rebuild the Chroma vector store from the markdown manuals in ../data/manuals.

Run from the backend/ directory:
    python scripts/ingest.py
"""
import glob
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "manuals"
VECTOR_STORE_DIR = Path(__file__).resolve().parent.parent / "data" / "vector_store"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
COLLECTION_NAME = "appliance_manuals"
OLLAMA_MODEL = "llama3.2"


def chunk_text(text, source, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Section-aware chunking: split on markdown headers first, then
    fixed-size-with-overlap within any section that's still too long."""
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    chunks = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        if len(sec) <= chunk_size:
            chunks.append(sec)
        else:
            start = 0
            while start < len(sec):
                end = start + chunk_size
                chunks.append(sec[start:end])
                start = end - overlap
    return [{"source": source, "chunk": c} for c in chunks]


def main():
    files = sorted(glob.glob(str(DATA_DIR / "*.md")))
    print(f"Found {len(files)} manual files:")
    for f in files:
        print(" -", f)

    docs = []
    for f in files:
        text = Path(f).read_text(encoding="utf-8")
        docs.append({"source": Path(f).name, "text": text})

    all_chunks = []
    for d in docs:
        all_chunks.extend(chunk_text(d["text"], d["source"]))
    print(f"Total chunks: {len(all_chunks)}")

    print(f"Loading embedder: {EMBED_MODEL_NAME}")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [c["chunk"] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    metadatas = [{"source": c["source"]} for c in all_chunks]

    existing = collection.get()["ids"]
    if existing:
        collection.delete(ids=existing)

    collection.add(
        ids=ids,
        embeddings=[e.tolist() for e in embeddings],
        documents=texts,
        metadatas=metadatas,
    )
    print(f"Persisted {collection.count()} chunks to {VECTOR_STORE_DIR}")

    import json

    config = {
        "embedding_model": EMBED_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "vector_store_path": str(VECTOR_STORE_DIR),
        "collection_name": COLLECTION_NAME,
        "ollama_model": OLLAMA_MODEL,
        "num_chunks": len(all_chunks),
    }
    with open(VECTOR_STORE_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Exported config:", config)


if __name__ == "__main__":
    main()
