import argparse
import mimetypes

from rag_search_engine.core.llm import image_describer

def main() -> None:
    parser = argparse.ArgumentParser(description="Describe Image CLI")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--query", help="Query to describe the image")

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or  "image/jpeg"

    description, tokens = image_describer(mime, args.image, args.query or "")
    print(f"Rewritten query: {description}")
    print(f"Total tokens: {tokens}")

if __name__ == "__main__":
    main()