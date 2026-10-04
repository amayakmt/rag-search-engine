import argparse
from utils.data import load_test_cases, load_movies
from core.hybrid_search import HybridSearch
from config import RRF_SEARCH_K

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    print(f"=== Evaluating Search (k={limit}) ===\n")

    hybrid = HybridSearch(load_movies())
    test_cases = load_test_cases()

    for case in test_cases:
        query = case["query"]
        expected_docs = case["relevant_docs"]

        results = hybrid.rrf_search(query, k=RRF_SEARCH_K, limit=limit)

        total_retrieved = len(results)
        total_relevant_available = len(expected_docs)
        relevant_retrieved = sum(1 for result in results if result.get("title") in expected_docs)

        precision_score = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0.0
        recall_score = relevant_retrieved / total_relevant_available if total_relevant_available > 0 else 0.0
        f1_score = (2 * (precision_score * recall_score) / (precision_score + recall_score)) if precision_score or recall_score else 0.0

        retrieved_titles = ", ".join([result.get("title") for result in results])
        expected_titles = ", ".join(expected_docs)

        print(f"- Query: '{query}'")
        print(f"    - Precision@{limit}: {precision_score:.4f}")
        print(f"    - Recall@{limit}: {recall_score:.4f}")
        print(f"    - F1 Score: {f1_score:.4f}")
        print(f"    - Retrieved: {retrieved_titles}")
        print(f"    - Relevant: {expected_titles}\n")

if __name__ == "__main__":
    main()