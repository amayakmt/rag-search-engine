import argparse
import json
import sys

from process_text import tokenize_text
from load_movies import load_movies
from build_command import build_command
from inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build and cache the inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            # tokenize query
            keyword_query = args.query
            tokenized_query = tokenize_text(keyword_query)

            # load the index
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)

            # initiate search
            print(f"Searching for: {keyword_query}")
            result = []

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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()