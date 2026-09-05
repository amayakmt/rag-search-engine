import argparse

from lib.semantic_search import SemanticSearch

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # verify model
    verify_parser = subparsers.add_parser("verify", help="verifies the model")


    args = parser.parse_args()

    match args.command:
        case "verify":
            model = SemanticSearch()
            model.verify_model()


        case _:
            parser.print_help()


if __name__ == "__main__":
    main()