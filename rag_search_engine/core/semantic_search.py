import hashlib
import json
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_search_engine.config import CACHE_DIR, EMBEDDINGS_PATH, CHUNKS_EMBEDDINGS_PATH, CHUNKS_METADATA_PATH
from rag_search_engine.utils.chunking import chunk_by_text_sentences
from rag_search_engine.utils.data import format_search_result

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings: np.ndarray | None = None
        self.documents: list[dict] | None = None
        self.document_map: dict[int, dict] = {}

    def _cache_fingerprint(self, documents: list[dict]) -> str:
        payload = {
            "model": self.model_name,
            "documents": [
                {"id": doc["id"], "title": doc["title"], "description": doc.get("description", "")}
                for doc in documents
            ],
            "chunk_size": 4,
            "overlap": 1,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()))
        return self.model.encode(texts, show_progress_bar=True)

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
        stringed_docs = [f"{doc['title']}: {doc.get('description', '')}" for doc in documents]
        
        self.embeddings = self._encode_documents(stringed_docs)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDINGS_PATH, self.embeddings)
        with open(EMBEDDINGS_PATH.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump({"fingerprint": self._cache_fingerprint(documents)}, f)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        metadata_path = EMBEDDINGS_PATH.with_suffix(".json")
        if EMBEDDINGS_PATH.is_file() and metadata_path.is_file():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            if metadata.get("fingerprint") == self._cache_fingerprint(documents):
                embeddings = np.load(EMBEDDINGS_PATH)
                if embeddings.ndim == 2 and len(embeddings) == len(documents):
                    self.embeddings = embeddings
                    return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query: str, limit: int) -> list[dict]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        if limit == 0 or not self.documents:
            return []
        query_embedding = self.generate_embedding(query)
        
        scored_docs = [
            (cosine_similarity(query_embedding, doc_emb), doc)
            for doc, doc_emb in zip(self.documents, self.embeddings)
        ]
        scored_docs.sort(key=lambda item: item[0], reverse=True)
        
        return [
            {
                "id": doc["id"],
                "score": score,
                "title": doc["title"],
                "description": doc.get("description", "")
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
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks)
                })

        self.chunk_embeddings = self._encode_documents(all_chunks)
        self.chunk_metadata = chunk_metadata

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(CHUNKS_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNKS_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "chunks": chunk_metadata,
                "total_chunks": len(all_chunks),
                "fingerprint": self._cache_fingerprint(documents),
            }, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        if CHUNKS_EMBEDDINGS_PATH.is_file() and CHUNKS_METADATA_PATH.is_file():
            with open(CHUNKS_METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            if metadata.get("fingerprint") == self._cache_fingerprint(documents):
                embeddings = np.load(CHUNKS_EMBEDDINGS_PATH)
                chunks = metadata.get("chunks", [])
                if embeddings.ndim == 2 and len(embeddings) == len(chunks):
                    self.chunk_embeddings = embeddings
                    self.chunk_metadata = chunks
                    return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("No chunk embeddings loaded. Call 'load_or_create_chunk_embeddings' first.")
        if limit == 0 or not self.documents or len(self.chunk_embeddings) == 0:
            return []

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
                description=doc.get("description", ""),
                score=best_chunk["score"],
                chunk_idx=best_chunk["chunk_idx"],
                movie_idx=best_chunk["movie_idx"]
            )
            results.append(formatted_result)

        return results