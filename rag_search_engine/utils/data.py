import json
from typing import Any, TypedDict
from rag_search_engine.config import MOVIES, SCORE_PRECISION, EVAL_DATASET

class SearchResult(TypedDict):
    id: int
    title: str
    description: str
    score: float
    metadata: dict[str, Any]

def format_search_result(
    doc_id: int, title: str, description: str, score: float, **metadata: Any
) -> SearchResult:
    return {
        "id": doc_id,
        "title": title,
        "description": description,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }

def load_movies() -> list[dict]:
    with open(MOVIES, "r", encoding="utf-8") as m:
        data = json.load(m)
        return data["movies"]

def load_test_cases() -> list[dict]:
    with open(EVAL_DATASET, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data["test_cases"]