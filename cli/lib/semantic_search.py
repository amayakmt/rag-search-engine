from sentence_transformers import SentenceTransformer
import numpy as np

from config import EMBEDDINGS_PATH

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}


    def verify_model(self):
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")


    def generate_embedding(self, text: str):
        stripped_text = text.strip()

        if not stripped_text:
            raise ValueError("Text must not be empty or contain only whitespaces")

        embeddings = self.model.encode([stripped_text])

        return embeddings[0]


    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        stringed_docs = []

        for doc in documents:
            id = doc["id"]
            description = doc["description"]
            title = doc["title"]

            self.document_map[id] = doc
            stringed = f"{title}: {description}"
            stringed_docs.append(stringed)

        self.embeddings = self.model.encode(stringed_docs, show_progress_bar=True)
        np.save(EMBEDDINGS_PATH, self.embeddings)

        return self.embeddings


    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        if EMBEDDINGS_PATH.is_file():
            self.embeddings = np.load(EMBEDDINGS_PATH)

            if len(self.embeddings) == len(documents):
                return self.embeddings
            else:
                raise ValueError("The length of the loaded embeddings does not match to the length of the documents.")

        else:
            return self.build_embeddings(documents)

    
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)

        # 1. Pair each document with its cosine similarity score
        scored_docs = []
        for doc, doc_emb in zip(self.documents, self.embeddings):
            score = cosine_similarity(query_embedding, doc_emb)
            scored_docs.append((score, doc))

        # 2. Sort descending by similarity score
        scored_docs.sort(key=lambda item: item[0], reverse=True)

        # 3. Format the top limit results into dictionaries
        results = []
        for score, doc in scored_docs[:limit]:
            results.append({
                "score": score,
                "title": doc["title"],
                "description": doc["description"]
            })

        return results


# ---
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
