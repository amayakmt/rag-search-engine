import json
from typing import Any, TypedDict
from config import MOVIES, SCORE_PRECISION

class SearchResult(TypedDict):
    id: int
    title: str
    document: str
    score: float
    metadata: dict[str, Any]

def format_search_result(
    doc_id: int, title: str, document: str, score: float, **metadata: Any
) -> SearchResult:
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }

def load_movies() -> dict:
    with open(MOVIES, "r") as m:
        data = json.load(m)
        return data["movies"]