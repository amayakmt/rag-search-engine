import argparse
import sys

from utils.data import load_movies
from core.hybrid_search import HybridSearch
from core.llm import spell_checker, rewriter, expand, individual_reranker, batch_reranker
from core.cross_encoder import cross_encoder_reranker

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
    rrf_search_command.add_argument("--enhance", type=str, nargs="?", const=None, choices=["spell", "rewrite", "expand"], default=None, help="query enhancement method")
    rrf_search_command.add_argument("--rerank-method", type=str, nargs="?", const=None, choices=["individual", "batch", "cross_encoder"], default=None, help="optional argument to implement re-ranking")

    args = parser.parse_args()

    # exit if no command is provided
    if not args.command:
        parser.print_help()
        sys.exit(0)

    # load resources only if we are actually seraching
    hybrid = None
    if args.command in ("weighted-search", "rrf-search"):
        hybrid = HybridSearch(load_movies())

    match args.command:
        case "normalize":
            dummy_hybrid = HybridSearch([])
            result = dummy_hybrid.normalize(args.scores)
            for score in result:
                print(f"* {score:.4f}")

        case "weighted-search":
            results = hybrid.weighted_search(args.query, alpha=args.alpha, limit=args.limit)
            for idx, result in enumerate(results, start=1):
                print(f"{idx}. {result['title']}")
                print(f"Hybrid Score: {result['hybrid_score']:.3f}")
                print(f"BM25: {result['bm25_score']:.3f}, Semantic: {result['semantic_score']:.3f}")
                print(f"{result['description']}...\n")

        case "rrf-search":
            query = args.query
            original_limit = args.limit
            limit = original_limit

            if args.enhance:
                enhancers = {
                    "spell": spell_checker,
                    "rewrite": rewriter,
                    "expand": expand
                }
                try:
                    enhanced_query = enhancers[args.enhance](query)["response"]
                    print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'")
                    query = enhanced_query
                except Exception as e:
                    print(f"Error: {e}")
                    sys.exit(1)

            if args.rerank_method:
                limit = original_limit * 5

            results = hybrid.rrf_search(query=query, k=args.k, limit=limit)

            rerankers = {
                "individual": individual_reranker,
                "batch": batch_reranker,
                "cross_encoder": cross_encoder_reranker,
            }

            if args.rerank_method:
                if args.rerank_method in rerankers:
                    print(f"Re-ranking top {original_limit} results using '{args.rerank_method}' method...")
                    results = rerankers[args.rerank_method](query, results)
                    results = results[:original_limit]
                else:
                    print(f"Error: Unknown re-rank method '{args.rerank_method}'")
                    sys.exit(1)

            print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")

            for idx, result in enumerate(results, start=1):
                print(f"{idx}. {result.get('title', 'Untitled')}")

                if "individual_rerank_score" in result:
                    print(f"    Re-rank Score: {result['individual_rerank_score']:.3f}/10")
                elif "batch_rerank_position" in result:
                    print(f"    Re-rank Rank: {result['batch_rerank_position']}")
                elif "cross_encoder_rerank_score":
                    print(f"    Cross-Encoder Score: {result['cross_encoder_rerank_score']:.3f}")

                print(f"    RRF Score: {result['rrf_score']:.3f}")
                print(f"    BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}")
                print(f"    {result['description']}...\n")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()