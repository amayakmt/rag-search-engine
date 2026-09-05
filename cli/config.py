from pathlib import Path

# base dir path
BASE_DIR = Path(__file__).resolve().parent.parent

# cache
CACHE_DIR = Path("cache")
INDEX_PATH = CACHE_DIR / "index.pkl"
DOCMAP_PATH = CACHE_DIR / "docmap.pkl"
TF_PATH = CACHE_DIR / "term_frequencies.pkl"
DOC_LENGTH_PATH = CACHE_DIR / "doc_lengths.pkl"
EMBEDDINGS_PATH = CACHE_DIR / "movie_embeddings.npy"

# data
STOPWORDS = BASE_DIR / "data" / "stopwords.txt"
MOVIES = BASE_DIR / "data" / "movies.json"

# constants
BM25_K1 = 1.5
BM25_B = 0.75
SEARCH_LIMIT = 5