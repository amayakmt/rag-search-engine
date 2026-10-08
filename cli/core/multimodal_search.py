from PIL import Image
from sentence_transformers import SentenceTransformer
from core.semantic_search import cosine_similarity

from config import MULTIMODAL_MODEL

class MultimodalSearch:
    def __init__(self, model_name: str = MULTIMODAL_MODEL, documents: list = None):
        self.model = SentenceTransformer(model_name)
        self.documents = documents if documents is not None else []
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image: str) -> list:
        img = Image.open(image)
        embedding = self.model.encode([img])[0]
        return embedding

    def verify_image_embedding(image: str) -> None:
        search = MultimodalSearch()
        embedding = search.embed_image(image)
        print(f"Embedding shape: {embedding.shape[0]} dimensions")

    def search_with_image(self, image: str):
        embedding = self.embed_image(image)
        similarities = [
            cosine_similarity(embedding, text_embedding)
            for text_embedding in self.text_embeddings
            ]
        
        sorted_docs = [
            {
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"],
                "similarity": sim
            }
            for sim, doc in sorted(zip(similarities, self.documents), key=lambda x: x[0], reverse=True)
        ][:5]
        return sorted_docs

def image_search_command(image: str, documents:list = None):
    search = MultimodalSearch(documents=documents)
    results = search.search_with_image(image)

    return results