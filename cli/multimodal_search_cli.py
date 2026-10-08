import argparse

from core.multimodal_search import MultimodalSearch

def main() -> None:
    parser = argparse.ArgumentParser(description="Describe Image CLI")
    subparser = parser.add_subparsers(dest="command")

    verify_image_embedding_parser = subparser.add_parser("verify_image_embedding", help="Verify image embedding")
    verify_image_embedding_parser.add_argument("image_path", help="Path to the image")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            MultimodalSearch.verify_image_embedding(args.image_path)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()