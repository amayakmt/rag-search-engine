# file to build an inverted index for keyword search
import pickle
from process_text import tokenize_text
from load_movies import load_movies
from config import CACHE_DIR, INDEX_PATH, DOCMAP_PATH

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {} # dict mapping tokens to sets of document IDs
        self.docmap = {} # dict mapping document IDs to their full document objects

    # add each movie token as a key, and map to a set of document IDs
    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    # get the document ID for a single preprocessed token and return them as a sorted list
    def get_documents(self, term):
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

    # loads the index and docmap from disk using pickle module's load function
    def load(self):
        if not INDEX_PATH.is_file() or not DOCMAP_PATH.is_file():
            raise FileNotFoundError("Cache files not found. Build the index first.")

        with open(INDEX_PATH, "rb") as f:
            self.index = pickle.load(f)

        with open(DOCMAP_PATH, "rb") as f:
            self.docmap = pickle.load(f)