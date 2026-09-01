# file to build an inverted index for keyword search
import pickle
from collections import Counter
import math

from tokenizer import tokenize_text
from load_movies import load_movies
from config import CACHE_DIR, INDEX_PATH, DOCMAP_PATH, TF_PATH, BM25_K1


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {} # dict mapping tokens to sets of document IDs
        self.docmap = {} # dict mapping document IDs to their full document objects
        self.term_frequencies: dict[str, Counter] = {} # dictionary mapping document IDs to Counter objects (dictionary optimized for counting)

    # add each movie token as a key, and map to a set of document IDs
    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    # get the document ID for a single preprocessed token and return them as a sorted list
    def get_documents(self, term) -> set:
        return sorted(self.index.get(term, set()))

    # add each movie to both the index and the docmap
    # uses each movie's existing ID as its document ID
    def build(self):
        movies = load_movies()

        for movie in movies:
            doc_id = movie["id"]

            # concatenate title and description and delegate tokenization to __add_documents
            text = f"{movie["title"]} {movie["description"]}"
            self.__add_document(doc_id, text)

            # add full movie to docmap
            self.docmap[doc_id] = movie



    # save the index and docmap attributes to disk using the pickle module's dump function
    def save(self):
        # creates "cache/" iif it doesn't already exist; ignores if it does
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        with open(INDEX_PATH, "wb") as f:
            pickle.dump(self.index, f)

        with open(DOCMAP_PATH, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(TF_PATH, "wb") as f:
            pickle.dump(self.term_frequencies, f)


    # loads the index and docmap from disk using pickle module's load function
    def load(self):
        if not INDEX_PATH.is_file() or not DOCMAP_PATH.is_file() or not TF_PATH.is_file():
            raise FileNotFoundError("Cache files not found. Build the index first.")

        with open(INDEX_PATH, "rb") as f:
            self.index = pickle.load(f)

        with open(DOCMAP_PATH, "rb") as f:
            self.docmap = pickle.load(f)

        with open(TF_PATH, "rb") as f:
            self.term_frequencies = pickle.load(f)


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
        doc_ids_with_term = self.get_documents(term)
        df = len(doc_ids_with_term)

        return math.log((total_doc_count - df + 0.5) / (df + 0.5) + 1)
    
    # returns BM25 enhanced tf
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1):
        raw_tf = self.get_tf(doc_id, term)
        return (raw_tf * (k1 + 1)) / (raw_tf + k1)