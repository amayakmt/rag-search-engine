import argparse
import sys

from core.tokenizer import tokenize_text, tokenize_term
from core.inverted_index import InvertedIndex
from config import BM25_K1, BM25_B, SEARCH_LIMIT

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build and cache the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Compute Term Frequency in a selected Document ID")
    tf_parser.add_argument("doc_id", type=int, help="Document ID to check")
    tf_parser.add_argument("term", help="Term to check")

    idf_parser = subparsers.add_parser("idf", help="Compute Inverse Document Frequency for a specific term")
    idf_parser.add_argument("term", help="Term to check")

    tfidf_parser = subparsers.add_parser("tfidf", help="Compure TF-IDF for a specific term and a document ID")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID to check")
    tfidf_parser.add_argument("term", help="Term to check")

    bm25idf_parser = subparsers.add_parser("bm25idf", help="Compute BM25 IDF")
    bm25idf_parser.add_argument("term", help="Term to check")

    bm25tf_parser = subparsers.add_parser("bm25tf", help="Compute BM25 TF")
    bm25tf_parser.add_argument("doc_id", type=int, help="Document ID to check")
    bm25tf_parser.add_argument("term", help="Term to check")
    bm25tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search via BM25")
    bm25search_parser.add_argument("query", help="Query to search for")
    bm25search_parser.add_argument("--limit", type=int, default=SEARCH_LIMIT, help="Maximum number of results to return")

    args = parser.parse_args()
    idx = InvertedIndex()

    match args.command:
        case "search":
            keyword_query = args.query
            tokenized_query = tokenize_text(keyword_query)

            try:
                idx.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)

            print(f"Searching for: {keyword_query}")

            doc_ids = set()
            for token in tokenized_query:
                for doc_id in idx.get_documents(token):
                    if doc_id not in doc_ids:
                        doc_ids.add(doc_id)
                    if len(doc_ids) == 5:
                        break
                if len(doc_ids) == 5:
                    break

            for i, doc_id in enumerate(doc_ids, start=1):
                movie = idx.docmap[doc_id]
                print(f"{i}. ({movie['id']}) {movie['title']}")

        case "build":
            idx.build()
            idx.save()
            print("Index built and saved successfully.")

        case "tf": 
            try:
                idx.load()
                tokenized_term = tokenize_term(args.term)
            except (FileNotFoundError, ValueError) as e:
                print(f"Error: {e}")
                sys.exit(1)

            print(idx.get_tf(args.doc_id, tokenized_term))

        case "idf":
            try:
                idx.load()
                tokenized_term = tokenize_term(args.term)
            except (FileNotFoundError, ValueError) as e:
                print(f"Error: {e}")
                sys.exit(1)

            idf = idx.get_idf(tokenized_term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            try:
                idx.load()
                tokenized_term = tokenize_term(args.term)
            except (FileNotFoundError, ValueError) as e:
                print(f"Error: {e}")
                sys.exit(1)

            tf_idf = idx.get_tfidf(args.doc_id, tokenized_term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            try:
                idx.load()
                tokenized_term = tokenize_term(args.term)
            except (FileNotFoundError, ValueError) as e:
                print(f"Error: {e}")
                sys.exit(1)

            bm25idf = idx.get_bm25_idf(tokenized_term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            try:
                idx.load()
                tokenized_term = tokenize_term(args.term)
            except (FileNotFoundError, ValueError) as e:
                print(f"Error: {e}")
                sys.exit(1)

            bm25tf = idx.get_bm25_tf(args.doc_id, tokenized_term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25search":
            try:
                idx.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)

            results = idx.bm25_search(args.query, args.limit)
            for i, movie in enumerate(results, start=1):
                print(f"{i}. ({movie['id']}) {movie['title']} - Score: {movie['score']:.2f}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()