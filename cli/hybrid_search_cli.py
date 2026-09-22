import argparse

from utils.data import load_movies
from core.hybrid_search import HybridSearch

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # normalize
    normalize_command = subparsers.add_parser("normalize", help="normalize the scores to 1.0 - 0.0 range")
    normalize_command.add_argument("scores", nargs="*", type=float, help="scores to normalize")

    # weighted search
    weighted_search_command = subparsers.add_parser("weighted-search", help="execute a hybrid search")
    weighted_search_command.add_argument("query", help="query to search")
    weighted_search_command.add_argument("--alpha", type=float, default=0.5, help="higher alpha gives more priority to keyword search")
    weighted_search_command.add_argument("--limit", type=int, default=5, help="maximum number of results to return")

    # reciprocal rank fusion search
    rrf_search_command = subparsers.add_parser("rrf-search", help="execute a rrf search")
    rrf_search_command.add_argument("query", help="query to search")
    rrf_search_command.add_argument("-k", type=int, default=60, help="a constant that controls how much more weight we give to higher-ranked results")
    rrf_search_command.add_argument("--limit", type=int, default=5, help="maximum number of results to return")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            hybrid = HybridSearch([])
            result = hybrid.normalize(args.scores)
            for score in result:
                print(f"* {score:.4f}")

        case "weighted-search":
            movies = load_movies()
            hybrid = HybridSearch(movies)
            results = hybrid.weighted_search(args.query, alpha=args.alpha, limit=args.limit)
            for idx, result in enumerate(results, start=1):
                print(f"{idx}. {result['title']}")
                print(f"Hybrid Score: {result['hybrid_score']:.3f}")
                print(f"BM25: {result['bm25_score']:.3f}, Semantic: {result['semantic_score']:.3f}")
                print(f"{result['description']}...\n")

        case "rrf-search":
            movies = load_movies()
            hybrid = HybridSearch(movies)
            results = hybrid.rrf_search(args.query, k=args.k, limit=args.limit)
            for idx, result in enumerate(results, start=1):
                print(f"{idx}. {result['title']}")
                print(f"RRF Score: {result['rrf_score']:.3f}")
                print(f"BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}")
                print(f"{result['description']}...\n")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()