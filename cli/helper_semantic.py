from lib.semantic_search import SemanticSearch
from load_movies import load_movies
import re

def embed_text(text):
    model = SemanticSearch()
    embedding = model.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    model = SemanticSearch()
    documents = load_movies()

    embeddings = model.load_or_create_embeddings(documents)

    print(f"Number of docs: {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_query_text(query):
    model = SemanticSearch()
    embedding = model.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def chunk_text_by_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("Overlap must be strictly less than chunk_size")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), step)
    ]

def chunk_by_text_sentences(text: str, max_chunk_size: int, overlap: int) -> list[str]:
    if overlap >= max_chunk_size:
        raise ValueError("Overlap must be strictly less than chunk_size")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")

    stripped_text = text.strip()
    if not stripped_text:
        return []

    # Split on whitespace immediately preceded by sentence-ending punctuation (.!?)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", stripped_text) if s.strip()]
    if not sentences:
        return []

    step = max_chunk_size - overlap
    chunks = []
    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i : i + max_chunk_size]
        chunks.append(" ".join(chunk_sentences))

    return chunks