from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")


    def verify_model(self):
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")


    def generate_embedding(self, text: str):
        stripped_text = text.strip()

        if not stripped_text:
            raise ValueError("Text must not be empty or contain only whitespaces")

        embeddings = self.model.encode([stripped_text])

        return embeddings[0]
