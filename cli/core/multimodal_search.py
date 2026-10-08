from PIL import Image
from sentence_transformers import SentenceTransformer

from config import MULTIMODAL_MODEL

class MultimodalSearch:
    def __init__(self, model_name: str = MULTIMODAL_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image: str) -> list:
        img = Image.open(image)
        embedding = self.model.encode([img])[0]
        return embedding

    def verify_image_embedding(image: str) -> None:
        search = MultimodalSearch()
        embedding = search.embed_image(image)
        print(f"Embedding shape: {embedding.shape[0]} dimensions")