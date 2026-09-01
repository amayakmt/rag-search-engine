from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CACHE_DIR = Path("cache")
INDEX_PATH = CACHE_DIR / "index.pkl"
DOCMAP_PATH = CACHE_DIR / "docmap.pkl"
TF_PATH = CACHE_DIR / "term_frequencies.pkl"

STOPWORDS = BASE_DIR / "data" / "stopwords.txt"
MOVIES = BASE_DIR / "data" / "movies.json"

BM25_K1 = 1.5