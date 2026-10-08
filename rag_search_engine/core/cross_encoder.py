from sentence_transformers import CrossEncoder
from rag_search_engine.config import CROSS_ENCODER_MODEL

def cross_encoder_reranker(query: str, documents: list[dict]) -> list[dict]:
    if not documents:
        return []
    encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    pairs: list[list] = []

    for doc in documents:
        text = f"{doc.get('title', '')} - {doc.get('description', '')}"
        pairs.append([query, text])

    scores = encoder.predict(pairs)

    for doc, score in zip(documents, scores):
        doc["cross_encoder_rerank_score"] = float(score)

    return sorted(documents, key=lambda item: item["cross_encoder_rerank_score"], reverse=True)
