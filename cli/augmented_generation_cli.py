import argparse
from utils.data import load_movies
from core.hybrid_search import HybridSearch
from core.llm import (
    augmented_generator,
    summarizer,
    citations_generator,
    question_handler
)

from config import RRF_SEARCH_K

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summary_parser = subparsers.add_parser("summarize", help="Generate a summary for a query")
    summary_parser.add_argument("query", type=str, help="Query to generate summary for")
    summary_parser.add_argument("--limit", type=int, default=5, help="Number of search results to consider for summarization")

    citations_parser = subparsers.add_parser("citations", help="Generate citations for a query")
    citations_parser.add_argument("query", type=str, help="Query to generate citations for")
    citations_parser.add_argument("--limit", type=int, default=5, help="Number of search results to consider for citations")

    question_parser = subparsers.add_parser("question", help="Ask a question based on search results")
    question_parser.add_argument("query", type=str, help="Question to ask")
    question_parser.add_argument("--limit", type=int, default=5, help="Number of search results to consider for answering the question")

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

        case "summarize":
            query = args.query
            movies = load_movies()
            hybrid = HybridSearch(movies)
            results = hybrid.rrf_search(query, k=RRF_SEARCH_K, limit=args.limit)

            summary = summarizer(query, results)

            print("Search Results:")
            for result in results:
                print(f"- {result.get('title', 'Untitled')}")

            print("\nLLM Summary:")
            print(summary)

        case "citations":
            query = args.query
            movies = load_movies()
            hybrid = HybridSearch(movies)
            results = hybrid.rrf_search(query, k=RRF_SEARCH_K, limit=args.limit)

            citations = citations_generator(query, results)

            print("Search Results:")
            for result in results:
                print(f"- {result.get('title', 'Untitled')}")

            print("\nLLM Answer:")
            print(citations)

        case "question":
            query = args.query
            movies = load_movies()
            hybrid = HybridSearch(movies)
            results = hybrid.rrf_search(query, k=RRF_SEARCH_K, limit=args.limit)

            answer = question_handler(query, results)

            print("Search Results:")
            for result in results:
                print(f"- {result.get('title', 'Untitled')}")

            print("\nAnswer:")
            print(answer)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()