"""Public configuration facade over the existing project settings."""

import os
from pathlib import Path

from cli.config import (
    BM25_B,
    BM25_K1,
    CACHE_DIR,
    CHUNKS_EMBEDDINGS_PATH,
    CHUNKS_METADATA_PATH,
    CROSS_ENCODER_MODEL,
    DOC_LENGTH_PATH,
    DOCMAP_PATH,
    EMBEDDINGS_PATH,
    EVAL_DATASET,
    INDEX_PATH,
    LLM_BASE_URL,
    MODEL,
    MOVIES,
    MULTIMODAL_MODEL,
    RRF_SEARCH_K,
    SCORE_PRECISION,
    SEARCH_LIMIT,
    STOPWORDS,
    TF_PATH,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


DATA_DIR = _project_path(os.environ.get("RAG_DATA_DIR", Path(MOVIES).parent))
CACHE_DIR = _project_path(os.environ.get("RAG_CACHE_DIR", CACHE_DIR))
MOVIES = DATA_DIR / Path(MOVIES).name
STOPWORDS = DATA_DIR / Path(STOPWORDS).name
EVAL_DATASET = DATA_DIR / Path(EVAL_DATASET).name
INDEX_PATH = CACHE_DIR / Path(INDEX_PATH).name
DOCMAP_PATH = CACHE_DIR / Path(DOCMAP_PATH).name
TF_PATH = CACHE_DIR / Path(TF_PATH).name
DOC_LENGTH_PATH = CACHE_DIR / Path(DOC_LENGTH_PATH).name
EMBEDDINGS_PATH = CACHE_DIR / Path(EMBEDDINGS_PATH).name
CHUNKS_EMBEDDINGS_PATH = CACHE_DIR / Path(CHUNKS_EMBEDDINGS_PATH).name
CHUNKS_METADATA_PATH = CACHE_DIR / Path(CHUNKS_METADATA_PATH).name