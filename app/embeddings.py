import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from app.config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
)


_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.ClientAPI] = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _chroma_client


def get_or_create_collection() -> chromadb.Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # unit-norm so dot product == cosine sim
    )
    return np.asarray(embeddings, dtype=np.float32)


def embed_single(text: str) -> np.ndarray:
    model = get_model()
    emb = model.encode(
        [text],
        normalize_embeddings=True,
    )
    return np.asarray(emb[0], dtype=np.float32)


def index_documents(documents: List[Dict], embeddings: np.ndarray) -> None:
    collection = get_or_create_collection()

    # ChromaDB upsert in batches of 500 to stay under its internal limits
    batch = 500
    for start in range(0, len(documents), batch):
        end = min(start + batch, len(documents))
        batch_docs = documents[start:end]
        batch_embs = embeddings[start:end]

        collection.upsert(
            ids=[d["doc_id"] for d in batch_docs],
            embeddings=[e.tolist() for e in batch_embs],
            metadatas=[
                {"category": d["category"], "text_preview": d["text"][:500]}
                for d in batch_docs
            ],
            documents=[d["text"] for d in batch_docs],
        )

    print(f"Indexed {len(documents)} documents into ChromaDB.")


def search(query_embedding: np.ndarray, n_results: int = 10,
           where: Optional[dict] = None) -> dict:
    collection = get_or_create_collection()
    kwargs = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    return collection.query(**kwargs)
