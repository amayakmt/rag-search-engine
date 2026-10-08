import argparse
from utils.data import load_movies
from core.hybrid_search import HybridSearch
from core.llm import augmented_generator

from config import RRF_SEARCH_K

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            movies = load_movies()
            hybrid = HybridSearch(movies)
            results = hybrid.rrf_search(query, k=RRF_SEARCH_K, limit=5)

            answer = augmented_generator(query, results)

            print("Search Results:")
            for result in results:
                print(f"- {result.get('title', 'Untitled')}")

            print("\nRAG Response:")
            print(answer)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()