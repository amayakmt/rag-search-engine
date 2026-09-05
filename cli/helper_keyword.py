import sys

from inverted_index import InvertedIndex
from tokenizer import tokenize_term

from config import BM25_K1, BM25_B, SEARCH_LIMIT

# helper for bm25idf command
def bm25_idf_command(term) -> float:
    idx = InvertedIndex()

    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        tokenized_term = tokenize_term(term)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    return idx.get_bm25_idf(tokenized_term)


# helper for bm25tf command
def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    idx = InvertedIndex()

    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        tokenized_term = tokenize_term(term)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    return idx.get_bm25_tf(doc_id, tokenized_term, k1, b)


# helper for bm25tf-idf command
def bm25_search_command(query, limit=SEARCH_LIMIT):
    idx = InvertedIndex()

    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    results = idx.bm25_search(query, limit)

    for i, movie in enumerate(results, start=1):
        print(f"{i}. ({movie["id"]}) {movie["title"]} - Score: {movie["score"]:.2f}")
