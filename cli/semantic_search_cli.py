import argparse

from core.semantic_search import SemanticSearch, ChunkedSemanticSearch
from utils.chunking import chunk_by_text_sentences, chunk_text_by_words
from utils.data import load_movies
from config import SEARCH_LIMIT

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verifies the model")
    
    embed_text_parser = subparsers.add_parser("embed_text", help="Embeds a text")
    embed_text_parser.add_argument("text", help="Text to embed")
    
    subparsers.add_parser("verify_embeddings", help="Verify embeddings")
    
    embed_query_parser = subparsers.add_parser("embed_query", help="Embed the user's query")
    embed_query_parser.add_argument("query", help="Query to embed")
    
    search_parser = subparsers.add_parser("search", help="Execute semantic search")
    search_parser.add_argument("query", help="Query to search")
    search_parser.add_argument("--limit", type=int, default=SEARCH_LIMIT, help="Maximum number of results to return")
    
    chunk_parser = subparsers.add_parser("chunk", help="Splits long text into smaller pieces for embedding.")
    chunk_parser.add_argument("text", help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Maximum size of the chunk")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping words")
    
    sem_chunk_parser = subparsers.add_parser("semantic_chunk", help="Splits long text into small pieces on sentence boundaries.")
    sem_chunk_parser.add_argument("text", help="Text to chunk")
    sem_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Max chunk size in sentences")
    sem_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping sentences")
    
    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search movies using chunked embeddings")
    search_chunked_parser.add_argument("query", help="Query to search")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")
    
    subparsers.add_parser("embed_chunks", help="Embed chunks by sentences")

    args = parser.parse_args()

    match args.command:
        case "verify":
            model = SemanticSearch()
            model.verify_model()

        case "embed_text":
            model = SemanticSearch()
            embedding = model.generate_embedding(args.text)
            print(f"Text: {args.text}")
            print(f"First 3 dimensions: {embedding[:3]}")
            print(f"Dimensions: {embedding.shape[0]}")

        case "verify_embeddings":
            model = SemanticSearch()
            documents = load_movies()
            embeddings = model.load_or_create_embeddings(documents)
            print(f"Number of docs: {len(documents)}")
            print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

        case "embed_query":
            model = SemanticSearch()
            embedding = model.generate_embedding(args.query)
            print(f"Query: {args.query}")
            print(f"First 3 dimensions: {embedding[:3]}")
            print(f"Shape: {embedding.shape}")

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

        case "search_chunked":
            movies = load_movies()
            model = ChunkedSemanticSearch()
            model.load_or_create_chunk_embeddings(movies)
            results = model.search_chunks(args.query, args.limit)

            for i, result in enumerate(results, start=1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document']}...")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()