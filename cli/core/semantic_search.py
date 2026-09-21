import json
import numpy as np
from sentence_transformers import SentenceTransformer

from config import CACHE_DIR, EMBEDDINGS_PATH, CHUNKS_EMBEDDINGS_PATH, CHUNKS_METADATA_PATH
from utils.chunking import chunk_by_text_sentences
from utils.data import format_search_result

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


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
        return self.model.encode([stripped_text])[0]

    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        stringed_docs = [f"{doc['title']}: {doc['description']}" for doc in documents]
        
        self.embeddings = self.model.encode(stringed_docs, show_progress_bar=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        if EMBEDDINGS_PATH.is_file():
            self.embeddings = np.load(EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings
            raise ValueError("The length of the loaded embeddings does not match to the length of the documents.")
        return self.build_embeddings(documents)

    def search(self, query: str, limit: int) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        
        scored_docs = [
            (cosine_similarity(query_embedding, doc_emb), doc)
            for doc, doc_emb in zip(self.documents, self.embeddings)
        ]
        scored_docs.sort(key=lambda item: item[0], reverse=True)
        
        return [
            {"score": score, "title": doc["title"], "description": doc["description"]}
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
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks)
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
                self.chunk_metadata = json.load(f)["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("No chunk embeddings loaded. Call 'load_or_create_chunk_embeddings' first.")

        query_embedding = super().generate_embedding(query)
        chunk_score: list[dict] = []

        for chunk_emb, metadata in zip(self.chunk_embeddings, self.chunk_metadata):
            chunk_score.append({
                "chunk_idx": metadata["chunk_idx"],
                "movie_idx": metadata["movie_idx"],
                "score": cosine_similarity(query_embedding, chunk_emb)
            })

        best_movie_scores: dict[int, dict] = {}
        for chunk in chunk_score:
            m_idx = chunk["movie_idx"]
            if m_idx not in best_movie_scores or chunk["score"] > best_movie_scores[m_idx]["score"]:
                best_movie_scores[m_idx] = chunk

        sorted_best_movies = sorted(best_movie_scores.values(), key=lambda item: item["score"], reverse=True)
        top_movies = sorted_best_movies[:limit]

        results = []
        for best_chunk in top_movies:
            doc = self.documents[best_chunk["movie_idx"]]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc.get("description", "")[:100],
                score=best_chunk["score"],
                chunk_idx=best_chunk["chunk_idx"],
                movie_idx=best_chunk["movie_idx"]
            )
            results.append(formatted_result)

        return results