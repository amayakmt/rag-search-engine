import math
import pickle
from collections import Counter

from config import (
    BM25_B,
    BM25_K1,
    CACHE_DIR,
    DOC_LENGTH_PATH,
    DOCMAP_PATH,
    INDEX_PATH,
    SEARCH_LIMIT,
    TF_PATH,
)
from utils.data import load_movies
from core.tokenizer import tokenize_text

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict = {}
        self.term_frequencies: dict[str, Counter] = {}
        self.doc_lengths: dict[int, int] = {}
        self._avg_doc_length: float | None = None

    def __add_document(self, doc_id, text) -> None:
        tokens = tokenize_text(text)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        self.doc_lengths[doc_id] = len(tokens)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def _get_avg_doc_length(self) -> float:
        if self._avg_doc_length is None:
            if not self.doc_lengths:
                return 0.0
            self._avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        return self._avg_doc_length

    def get_documents(self, term) -> list[int]:
        return sorted(self.index.get(term, set()))

    def build(self) -> None:
        movies = load_movies()

        for movie in movies:
            doc_id = movie["id"]
            text = f"{movie['title']} {movie.get('description', '')}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie
            
        self._avg_doc_length = None

    def save(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        with open(INDEX_PATH, "wb") as f:
            pickle.dump(self.index, f)

        with open(DOCMAP_PATH, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(TF_PATH, "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(DOC_LENGTH_PATH, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        if not INDEX_PATH.is_file() or not DOCMAP_PATH.is_file() or not TF_PATH.is_file() or not DOC_LENGTH_PATH.is_file():
            raise FileNotFoundError("Cache files not found. Build the index first.")

        with open(INDEX_PATH, "rb") as f:
            self.index = pickle.load(f)

        with open(DOCMAP_PATH, "rb") as f:
            self.docmap = pickle.load(f)

        with open(TF_PATH, "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(DOC_LENGTH_PATH, "rb") as f:
            self.doc_lengths = pickle.load(f)
            
        self._avg_doc_length = None

    def get_tf(self, doc_id, term) -> float:
        return self.term_frequencies.get(doc_id, Counter())[term]

    def get_idf(self, term) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(term))
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_tfidf(self, doc_id, term):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        doc_ids_with_term = self.index.get(term, set())
        df = len(doc_ids_with_term)
        return math.log((total_doc_count - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
        raw_tf = self.get_tf(doc_id, term)
        avg_len = self._get_avg_doc_length()
        if raw_tf == 0 or avg_len == 0:
            return 0.0

        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_len)
        return (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)

    def bm25(self, doc_id, term) -> float:
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf

    def bm25_search(self, query, limit=SEARCH_LIMIT) -> list[dict]:
        tokenized_query = tokenize_text(query)
        if not tokenized_query:
            return []

        candidate_ids = set()
        for token in tokenized_query:
            candidate_ids.update(self.index.get(token, set()))

        scores = {}
        for doc_id in candidate_ids:
            scores[doc_id] = sum(self.bm25(doc_id, token) for token in tokenized_query)

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]

        return [
            {
                "id": doc_id,
                "title": self.docmap[doc_id]["title"],
                "description": self.docmap[doc_id].get("description", "")[:100],
                "score": score,
            }
            for doc_id, score in sorted_scores
        ]