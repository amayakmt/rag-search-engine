from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
from rag_search_engine.core.semantic_search import cosine_similarity

from rag_search_engine.config import MULTIMODAL_MODEL

class MultimodalSearch:
    def __init__(self, model_name: str = MULTIMODAL_MODEL, documents: list[dict] | None = None):
        self.model = SentenceTransformer(model_name)
        self.documents = documents if documents is not None else []
        self.texts = [f"{doc['title']}: {doc.get('description', '')}" for doc in self.documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True) if self.texts else []

    def embed_image(self, image: str) -> np.ndarray:
        with Image.open(image) as img:
            return self.model.encode([img])[0]

    @staticmethod
    def verify_image_embedding(image: str) -> None:
        search = MultimodalSearch()
        embedding = search.embed_image(image)
        print(f"Embedding shape: {embedding.shape[0]} dimensions")

    def search_with_image(self, image: str):
        if not self.documents:
            return []
        embedding = self.embed_image(image)
        similarities = [
            cosine_similarity(embedding, text_embedding)
            for text_embedding in self.text_embeddings
            ]
        
        sorted_docs = [
            {
                "id": doc["id"],
                "title": doc["title"],
                "description": doc.get("description", ""),
                "similarity": sim
            }
            for sim, doc in sorted(zip(similarities, self.documents), key=lambda x: x[0], reverse=True)
        ][:5]
        return sorted_docs

def image_search_command(image: str, documents: list[dict] | None = None) -> list[dict]:
    search = MultimodalSearch(documents=documents)
    results = search.search_with_image(image)

    return results