import argparse

from lib.semantic_search import SemanticSearch
from helper_semantic import embed_text

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # verify model
    verify_parser = subparsers.add_parser("verify", help="Verifies the model")

    # embed_text
    embed_text_parser = subparsers.add_parser("embed_text", help="Embeds a text")
    embed_text_parser.add_argument("text", help="Text to embed")

    args = parser.parse_args()

    match args.command:
        case "verify":
            model = SemanticSearch()
            model.verify_model()


        case "embed_text":
            text = args.text
            embed_text(text)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()