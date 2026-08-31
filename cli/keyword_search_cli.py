import argparse
import json
import sys

from tokenizer import tokenize_text, tokenize_term
from build_command import build_command
from inverted_index import InvertedIndex

def main() -> None:
    # CLI arguments logic
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build and cache the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Compute Term Frequency in a selected Document ID")
    tf_parser.add_argument("document_id", type=int, help="Document ID to check")
    tf_parser.add_argument("term", help="Term to check")

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
            doc_id = args.document_id
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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()