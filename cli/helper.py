import sys

from inverted_index import InvertedIndex
from tokenizer import tokenize_term

from config import BM25_K1

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

def bm25_tf_command(doc_id, term, k1=BM25_K1):
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

    return idx.get_bm25_tf(doc_id, tokenized_term, k1)