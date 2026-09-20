import argparse

from lib.semantic_search import (
    SemanticSearch,
    ChunkedSemanticSearch,
    chunk_by_text_sentences,
    chunk_text_by_words,
)
from helper_semantic import (
    embed_text,
    verify_embeddings,
    embed_query_text,
)
from load_movies import load_movies
from config import SEARCH_LIMIT


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # verify model
    subparsers.add_parser("verify", help="Verifies the model")

    # embed_text
    embed_text_parser = subparsers.add_parser("embed_text", help="Embeds a text")
    embed_text_parser.add_argument("text", help="Text to embed")

    # verify doc embeddings
    subparsers.add_parser("verify_embeddings", help="Verify embeddings")

    # embed user query
    embed_query_parser = subparsers.add_parser("embed_query", help="Embed the user's query")
    embed_query_parser.add_argument("query", help="Query to embed")

    # semantic search
    search_parser = subparsers.add_parser("search", help="Execute semantic search")
    search_parser.add_argument("query", help="Query to search")
    search_parser.add_argument("--limit", type=int, default=SEARCH_LIMIT, help="Maximum number of results to return")

    # chunking
    chunk_parser = subparsers.add_parser("chunk", help="Splits long text into smaller pieces for embedding.")
    chunk_parser.add_argument("text", help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Maximum size of the chunk")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping words")

    # semantic chunking
    sem_chunk_parser = subparsers.add_parser("semantic_chunk", help="Splits long text into small pieces on sentence boundaries.")
    sem_chunk_parser.add_argument("text", help="Text to chunk")
    sem_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Max chunk size in sentences")
    sem_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping sentences")

    # embed chunks
    subparsers.add_parser("embed_chunks", help="Embed chunks by sentences")

    args = parser.parse_args()

    match args.command:
        case "verify":
            model = SemanticSearch()
            model.verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embed_query":
            embed_query_text(args.query)

        case "search":
            model = SemanticSearch()
            movies = load_movies()
            model.load_or_create_embeddings(movies)
            results = model.search(args.query, args.limit)

            for idx, result in enumerate(results, start=1):
                print(f"{idx}. {result['title']} (score: {result['score']:.4f})")
                print(f"{result['description']}\n")

        case "chunk":
            chunks = chunk_text_by_words(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for idx, chunk in enumerate(chunks, start=1):
                print(f"{idx}. {chunk}")

        case "semantic_chunk":
            chunks = chunk_by_text_sentences(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for idx, chunk in enumerate(chunks, start=1):
                print(f"{idx}. {chunk}")

        case "embed_chunks":
            movies = load_movies()
            inst = ChunkedSemanticSearch()
            embeddings = inst.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
    