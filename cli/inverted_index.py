# file to build an inverted index for keyword search
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
from load_movies import load_movies
from tokenizer import tokenize_text


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}  # dict mapping tokens to sets of document IDs
        self.docmap: dict = {}  # dict mapping document IDs to their full document objects
        self.term_frequencies: dict[str, Counter] = {}  # dictionary mapping document IDs to Counter objects (dictionary optimized for counting)
        self.doc_lengths: dict[int, int] = {}
        self._avg_doc_length: float | None = None

    # add each movie token as a key, and map to a set of document IDs
    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        # cache the length of the document
        self.doc_lengths[doc_id] = len(tokens)

        # iterate over each token to store inverted index & compute term frequency per token
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    # calculate and return the average document length across all documents
    def __get_avg_doc_length(self) -> float:
        if self._avg_doc_length is None:
            if not self.doc_lengths:
                return 0.0
            self._avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        return self._avg_doc_length

    # get the document ID for a single preprocessed token and return them as a sorted list
    def get_documents(self, term) -> list[int]:
        return sorted(self.index.get(term, set()))

    # add each movie to both the index and the docmap
    # uses each movie's existing ID as its document ID
    def build(self):
        movies = load_movies()

        for movie in movies:
            doc_id = movie["id"]

            # concatenate title and description and delegate tokenization to __add_documents
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)

            # add full movie to docmap
            self.docmap[doc_id] = movie
        self._avg_doc_length = None

    # save cache to disk using the pickle module's dump function
    def save(self):
        # creates "cache/" if it doesn't already exist; ignores if it does
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        with open(INDEX_PATH, "wb") as f:
            pickle.dump(self.index, f)

        with open(DOCMAP_PATH, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(TF_PATH, "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(DOC_LENGTH_PATH, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    # loads the cache from disk using pickle module's load function
    def load(self):
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

    # returns simple tf
    def get_tf(self, doc_id, term):
        return self.term_frequencies.get(doc_id, Counter())[term]

    # returns simple idf
    def get_idf(self, term):
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(term))
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    # returns simple tf-idf
    def get_tfidf(self, doc_id, term):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    # returns BM25 enhanced idf
    def get_bm25_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        doc_ids_with_term = self.index.get(term, set())
        df = len(doc_ids_with_term)

        return math.log((total_doc_count - df + 0.5) / (df + 0.5) + 1)

    # returns BM25 enhanced tf
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        raw_tf = self.get_tf(doc_id, term)
        avg_len = self.__get_avg_doc_length()
        if raw_tf == 0 or avg_len == 0:
            return 0.0

        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_len)
        return (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)

    # return BM25 tf-idf
    def bm25(self, doc_id, term):
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)

        return tf * idf


    # search tool using BM25 tf-idf
    def bm25_search(self, query, limit=SEARCH_LIMIT):
        tokenized_query = tokenize_text(query)
        if not tokenized_query:
            return []

        # Find only matching candidate document IDs
        candidate_ids = set()
        for token in tokenized_query:
            candidate_ids.update(self.index.get(token, set()))

        # Score only candidate documents
        scores = {}
        for doc_id in candidate_ids:
            scores[doc_id] = sum(self.bm25(doc_id, token) for token in tokenized_query)

        # Sort candidate documents by score descending
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]

        # Return a unified list of results
        return [
            {
                "id": doc_id,
                "title": self.docmap[doc_id]["title"],
                "score": score,
            }
            for doc_id, score in sorted_scores
        ]
    