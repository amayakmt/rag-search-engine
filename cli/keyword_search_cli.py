import argparse
import sys

from tokenizer import tokenize_text, tokenize_term
from build_command import build_command
from inverted_index import InvertedIndex
from helper import bm25_idf_command, bm25_tf_command

from config import BM25_K1

def main() -> None:
    # CLI arguments logic
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Search
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    # Build
    build_parser = subparsers.add_parser("build", help="Build and cache the inverted index")

    # TF
    tf_parser = subparsers.add_parser("tf", help="Compute Term Frequency in a selected Document ID")
    tf_parser.add_argument("doc_id", type=int, help="Document ID to check")
    tf_parser.add_argument("term", help="Term to check")

    # IDF
    idf_parser = subparsers.add_parser("idf", help="Compute Inverse Document Frequency for a specific term")
    idf_parser.add_argument("term", help="Term to check")

    # TF-IDF
    tfidf_parser = subparsers.add_parser("tfidf", help="Compure TF-IDF for a specific term and a document ID")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID to check")
    tfidf_parser.add_argument("term", help="Term to check")

    # BM25 IDF
    bm25idf_parser = subparsers.add_parser("bm25idf", help="Compute BM25 IDF")
    bm25idf_parser.add_argument("term", help="Term to check")

    # BM25 TF
    bm25tf_parser = subparsers.add_parser("bm25tf", help="Compute BM25 TF")
    bm25tf_parser.add_argument("doc_id", type=int, help="Document ID to check")
    bm25tf_parser.add_argument("term", help="Term to check")
    bm25tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")

    args = parser.parse_args()

    # initialize inverted index
    idx = InvertedIndex()

    match args.command:
        case "search":
            # tokenize query
            keyword_query = args.query
            tokenized_query = tokenize_text(keyword_query)

            # load the index
            try:
                idx.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)

            # initiate search
            print(f"Searching for: {keyword_query}")

            # loop logic
            doc_ids = set()
            for token in tokenized_query:
                for doc_id in idx.get_documents(token):
                    if doc_id not in doc_ids:
                        doc_ids.add(doc_id)
                    if len(doc_ids) == 5:
                        break
                if len(doc_ids) == 5:
                    break

            # print logic
            for i, doc_id in enumerate(doc_ids, start=1):
                movie = idx.docmap[doc_id]
                print(f"f{i}. ({movie["id"]}) {movie["title"]}")

        case "build":
            build_command()

        case "tf": 
            doc_id = args.doc_id
            term = args.term

            # load the index
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

            print(idx.get_tf(doc_id, tokenized_term))

        case "idf":
            term = args.term

            # load the index
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

            idf = idx.get_idf(tokenized_term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            doc_id = args.doc_id
            term = args.term

            # load the index
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

            tf_idf = idx.get_tfidf(doc_id, tokenized_term)
            print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            term = args.term
            bm25idf = bm25_idf_command(term)
            print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")

        case "bm25tf":
            doc_id = args.doc_id
            term = args.term
            k1 = args.k1
            bm25tf = bm25_tf_command(doc_id, term, k1)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()