import argparse
import json
from process_text import tokenize_text
from load_movies import load_movies
from build_command import build_command

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

            # initiate
            print(f"Searching for: {keyword_query}")
            result = []

            movies = load_movies()

            for movie in movies:

                # tokenize movie
                tokenized_movie = tokenize_text(movie["title"])

                # check each query token to each movie token
                # allow partial matches
                match_found = any(
                    q_token in m_token
                    for q_token in tokenized_query
                    for m_token in tokenized_movie
                )

                if match_found:
                    result.append(movie["title"])

            # print out the search result
            for i, title in enumerate(result[:5], start=1):
                print(f"{i}. {title}")

            # truncate the search result to 5
            if len(result) > 5:
                print("...")

        case "build":
            build_command()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()