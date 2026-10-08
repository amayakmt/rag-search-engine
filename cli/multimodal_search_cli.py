import argparse

from core.multimodal_search import MultimodalSearch, image_search_command
from utils.data import load_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Describe Image CLI")
    subparser = parser.add_subparsers(dest="command")

    verify_image_embedding_parser = subparser.add_parser("verify_image_embedding", help="Verify image embedding")
    verify_image_embedding_parser.add_argument("image_path", help="Path to the image")

    image_search_parser = subparser.add_parser("image_search", help="Search with an image")
    image_search_parser.add_argument("image_path", help="Path to the image")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            MultimodalSearch.verify_image_embedding(args.image_path)

        case "image_search":
            results = image_search_command(args.image_path, documents=load_movies())
            for idx, result in enumerate(results, start=1):
                print(f"{idx}. {result['title']} (similarity: {result['similarity']:.3f})")
                print(f"    {result['description'][:100]}")
                
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()