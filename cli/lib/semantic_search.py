import json
import re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from config import CACHE_DIR, EMBEDDINGS_PATH, CHUNKS_EMBEDDINGS_PATH, CHUNKS_METADATA_PATH


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def chunk_text_by_words(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
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

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", stripped_text) if s.strip()]
    if not sentences:
        return []

    step = max_chunk_size - overlap
    chunks = []
    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i : i + max_chunk_size]

        if i > 0 and len(chunk_sentences) <= overlap:
            break

        chunks.append(" ".join(chunk_sentences))

    return chunks


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings: np.ndarray | None = None
        self.documents: list[dict] | None = None
        self.document_map: dict[int, dict] = {}

    def verify_model(self) -> None:
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")

    def generate_embedding(self, text: str) -> np.ndarray:
        stripped_text = text.strip()
        if not stripped_text:
            raise ValueError("Text must not be empty or contain only whitespaces")

        embeddings = self.model.encode([stripped_text])
        return embeddings[0]

    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        stringed_docs = [
            f"{doc['title']}: {doc['description']}"
            for doc in documents
        ]

        self.embeddings = self.model.encode(stringed_docs, show_progress_bar=True)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDINGS_PATH, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        if EMBEDDINGS_PATH.is_file():
            self.embeddings = np.load(EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings
            raise ValueError("The length of the loaded embeddings does not match to the length of the documents.")

        return self.build_embeddings(documents)

    def search(self, query, limit) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)

        scored_docs = [
            (cosine_similarity(query_embedding, doc_emb), doc)
            for doc, doc_emb in zip(self.documents, self.embeddings)
        ]

        scored_docs.sort(key=lambda item: item[0], reverse=True)

        return [
            {
                "score": score,
                "title": doc["title"],
                "description": doc["description"]
            }
            for score, doc in scored_docs[:limit]
        ]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings: np.ndarray | None = None
        self.chunk_metadata: list[dict] | None = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        all_chunks: list[str] = []
        chunk_metadata: list[dict] = []

        for movie_idx, doc in enumerate(documents):
            desc = doc.get("description", "").strip()
            if not desc:
                continue

            chunks = chunk_by_text_sentences(desc, max_chunk_size=4, overlap=1)
            total_chunks = len(chunks)

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks
                })

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(CHUNKS_EMBEDDINGS_PATH, self.chunk_embeddings)

        with open(CHUNKS_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        if CHUNKS_EMBEDDINGS_PATH.is_file() and CHUNKS_METADATA_PATH.is_file():
            self.chunk_embeddings = np.load(CHUNKS_EMBEDDINGS_PATH)
            with open(CHUNKS_METADATA_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)
