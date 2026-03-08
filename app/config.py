
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data" / "mini_newsgroups" / "mini_newsgroups"
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")
CLUSTER_ARTIFACTS_DIR = BASE_DIR / "artifacts"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

CHROMA_COLLECTION_NAME = "newsgroups"

MIN_K = 10
MAX_K = 30

DEFAULT_SIMILARITY_THRESHOLD = 0.85

MAX_CACHE_ENTRIES_PER_CLUSTER = 200
