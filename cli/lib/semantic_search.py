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
        