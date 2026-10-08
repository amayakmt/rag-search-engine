from .inverted_index import InvertedIndex

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank: int | None, k: int = 60) -> float:
    if k < 0:
        raise ValueError("k must be non-negative")
    if rank is None:
        return 0.0
    return 1 / (k + rank)


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        from .semantic_search import ChunkedSemanticSearch

        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        try:
            self.idx.load()
        except FileNotFoundError:
            self.idx.build(documents)
            self.idx.save()
        else:
            if self.idx.docmap != {doc["id"]: doc for doc in documents}:
                self.idx.build(documents)
                self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return []
        bm25_results: list[dict] = self._bm25_search(query, limit * 500)
        semantic_results: list[dict] = self.semantic_search.search_chunks(query, limit * 500)

        bm25_norm: list[float] = self.normalize([res["score"] for res in bm25_results])
        semantic_norm: list[float] = self.normalize([res["score"] for res in semantic_results])

        combined_results: dict[int, dict] = {}

        # add bm_25 normalized score
        for result, norm_score in zip(bm25_results, bm25_norm):
            doc_id = result["id"]
            combined_results[doc_id] = {
                "id": doc_id,
                "title": result["title"],
                "description": result["description"],
                "bm25_score": norm_score,
                "semantic_score": 0.0
            }

        # add semantic normalized score
        for result, norm_score in zip(semantic_results, semantic_norm):
            doc_id = result["id"]
            if doc_id in combined_results:
                combined_results[doc_id]["semantic_score"] = norm_score
            else:
                combined_results[doc_id] = {
                    "id": doc_id,
                    "title": result["title"],
                    "description": result["description"],
                    "bm25_score": 0.0,
                    "semantic_score": norm_score
                }

        for doc in combined_results.values():
            doc["hybrid_score"] = hybrid_score(doc["bm25_score"], doc["semantic_score"], alpha)

        return sorted(combined_results.values(), key=lambda item: item["hybrid_score"], reverse=True)[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        if k < 0:
            raise ValueError("k must be non-negative")
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return []
        bm25_results: list[dict] = self._bm25_search(query, limit * 500)
        semantic_results: list[dict] = self.semantic_search.search_chunks(query, limit * 500)

        combined_results: dict[int, dict] = {}

        for idx, result in enumerate(bm25_results, start=1):
            doc_id = result["id"]
            combined_results[doc_id] = {
                "id": doc_id,
                "title": result["title"],
                "description": result["description"],
                "bm25_rank": idx,
                "semantic_rank": None
            }

        for idx, result in enumerate(semantic_results, start=1):
            doc_id = result["id"]
            if doc_id in combined_results:
                combined_results[doc_id]["semantic_rank"] = idx
            else:
                combined_results[doc_id] = {
                    "id": doc_id,
                    "title": result["title"],
                    "description": result["description"],
                    "bm25_rank": None,
                    "semantic_rank": idx
                }

        for doc in combined_results.values():
            doc["rrf_score"] = (
                rrf_score(doc["bm25_rank"], k) +
                rrf_score(doc["semantic_rank"], k)
            )

        return sorted(combined_results.values(), key=lambda item: item["rrf_score"], reverse=True)[:limit]

    @staticmethod
    def normalize(scores: list[float]) -> list[float]:
        if not scores:
            return []

        max_s, min_s = max(scores), min(scores)

        if max_s == min_s:
            return [1.00] * len(scores)

        return [(score - min_s) / (max_s - min_s) for score in scores]
