import argparse
from rag_search_engine.utils.data import load_movies
from rag_search_engine.core.hybrid_search import HybridSearch
from rag_search_engine.core.llm import (
    augmented_generator,
    summarizer,
    citations_generator,
    question_handler
)

from rag_search_engine.config import RRF_SEARCH_K

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

    if not args.command:
        parser.print_help()
        return

    generators = {
        "rag": (augmented_generator, "RAG Response"),
        "summarize": (summarizer, "LLM Summary"),
        "citations": (citations_generator, "LLM Answer"),
        "question": (question_handler, "Answer"),
    }
    generator, heading = generators[args.command]
    hybrid = HybridSearch(load_movies())
    results = hybrid.rrf_search(args.query, k=RRF_SEARCH_K, limit=getattr(args, "limit", 5))
    answer = generator(args.query, results)

    print("Search Results:")
    for result in results:
        print(f"- {result.get('title', 'Untitled')}")
    print(f"\n{heading}:")
    print(answer)

if __name__ == "__main__":
    main()