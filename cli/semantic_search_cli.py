import argparse

from lib.semantic_search import SemanticSearch
from helper_semantic import embed_text, verify_embeddings, embed_query_text
from load_movies import load_movies

from config import SEARCH_LIMIT

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # verify model
    verify_parser = subparsers.add_parser("verify", help="Verifies the model")

    # embed_text
    embed_text_parser = subparsers.add_parser("embed_text", help="Embeds a text")
    embed_text_parser.add_argument("text", help="Text to embed")

    # verify doc embeddings
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify embeddings")

    # embed user query
    embed_query_parser = subparsers.add_parser("embed_query", help="Embed the user's query")
    embed_query_parser.add_argument("query", help="Query to embed")

    # semantic search
    search_parser = subparsers.add_parser("search", help="Execute semantic search")
    search_parser.add_argument("query", help="Query to search")
    search_parser.add_argument("--limit", type=int, default=SEARCH_LIMIT, help="Maximum number of results to return")

    args = parser.parse_args()

    match args.command:
        case "verify":
            model = SemanticSearch()
            model.verify_model()


        case "embed_text":
            text = args.text
            embed_text(text)


        case "verify_embeddings":
            verify_embeddings()


        case "embed_query":
            query = args.query
            embed_query_text(query)


        case "search":
            query = args.query
            limit = args.limit

            model = SemanticSearch()
            movies = load_movies()

            model.load_or_create_embeddings(movies)
            results = model.search(query, limit)

            for idx, result in enumerate(results, start=1):
                print(f"{idx}. {result["title"]} (score: {result["score"]:.4f})")
                print(f"{result["description"]}\n")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()