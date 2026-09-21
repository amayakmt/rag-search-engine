import argparse

from core.hybrid_search import HybridSearch

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # normalize
    normalize_command = subparsers.add_parser("normalize", help="normalize the scores to 1.0 - 0.0 range")
    normalize_command.add_argument("scores", nargs="*", type=float, help="scores to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            idx = HybridSearch([])
            result = idx.normalize(args.scores)
            for score in result:
                print(f"* {score:.4f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()