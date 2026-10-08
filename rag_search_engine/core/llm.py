import os
import time
import json
from typing import TypedDict
from dotenv import load_dotenv
from openai import OpenAI
import base64

from rag_search_engine.config import MODEL, LLM_BASE_URL
from rag_search_engine.core.llm_prompts import (
    AUGMENTED_GENERATION,
    IMAGE_DESCRIBER,
    SPELL_CHECKER,
    REWRITER,
    EXPANSION,
    INDIVIDUAL_RERANK,
    BATCH_RERANK,
    EVALUATE,
    SUMMARIZER,
    CITATIONS,
    QUESTION
)

class LLMResponse(TypedDict):
    response: str
    prompt_tokens: int
    response_tokens: int

_client: OpenAI | None = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        load_dotenv()
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set. Please set it in your .env file.")
        _client = OpenAI(base_url=LLM_BASE_URL, api_key=api_key)
    return _client

def invoke_llm(prompt: str) -> LLMResponse:
    client = get_client()
    messages = [{"role": "user", "content": prompt}]

    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )

    return {
        "response": completion.choices[0].message.content or "",
        "prompt_tokens": completion.usage.prompt_tokens if completion.usage else 0,
        "response_tokens": completion.usage.completion_tokens if completion.usage else 0,
    }

# --- Query Enhancers ---

def spell_checker(query: str) -> LLMResponse:
    return invoke_llm(f"{SPELL_CHECKER} '{query}'")

def rewriter(query: str) -> LLMResponse:
    return invoke_llm(f"{REWRITER} '{query}'")

def expand(query: str) -> LLMResponse:
    return invoke_llm(f"{EXPANSION} '{query}'")

# --- Reranking ---

def individual_reranker(query: str, documents: list[dict]) -> list[dict]:
    for doc in documents:
        prompt = INDIVIDUAL_RERANK.format(
            query=query,
            title=doc.get("title", ""),
            document=doc.get("description", "")
        )
        raw_score = invoke_llm(prompt)["response"]

        try:
            doc["individual_rerank_score"] = float(raw_score.strip())
        except ValueError:
            doc["individual_rerank_score"] = 0.0

        time.sleep(3)

    return sorted(documents, key=lambda item: item["individual_rerank_score"], reverse=True)

def batch_reranker(query: str, documents: list[dict]) -> list[dict]:
    doc_lines = []
    for doc in documents:
        doc_id = doc["id"]
        title = doc.get("title", "")
        desc = doc.get("description", "")
        doc_lines.append(f"ID: {doc_id} | Title: {title} | Desc: {desc}")

    doc_list_str = "\n".join(doc_lines)

    prompt = BATCH_RERANK.format(
        query=query,
        doc_list_str=doc_list_str
    )

    raw_response = invoke_llm(prompt)["response"]

    try:
        ranked_ids = json.loads(raw_response)
        if not isinstance(ranked_ids, list) or any(type(doc_id) is not int for doc_id in ranked_ids):
            raise ValueError("Expected a JSON list of document IDs")
    except (json.JSONDecodeError, ValueError):
        print("Warning: LLM failed to return valid JSON. Falling back to original RRF ranking.")
        return documents

    rank_map = {doc_id: rank for rank, doc_id in enumerate(ranked_ids)}

    ranked_documents = sorted(documents, key=lambda doc: rank_map.get(doc["id"], float('inf')))

    for idx, doc in enumerate(ranked_documents, start=1):
        doc["batch_rerank_position"] = idx

    return ranked_documents

def _format_results(results: list[dict]) -> str:
    return "\n".join(
        f"Result {idx}: {result.get('title', 'Untitled')} - {result.get('description', '')}"
        for idx, result in enumerate(results, start=1)
    )


def evaluator(query: str, results: list[dict]) -> list[int]:
    if not results:
        return []

    prompt = EVALUATE.format(
        query=query,
        doc_list_str=_format_results(results)
    )

    try:
        raw_response = invoke_llm(prompt)["response"].strip()
    except Exception as e:
        print(f"Warning: LLM invocation failed for evaluator ({e}). Defaulting to 0s.")
        return [0] * len(results)

    try:
        if raw_response.startswith("```"):
            raw_response = raw_response.replace("```json", "").replace("```", "").strip()

        scores = json.loads(raw_response)

        if not isinstance(scores, list):
            raise ValueError("Expected a JSON list of scores")
        scores = [int(score) for score in scores]
        if any(score < 0 or score > 3 for score in scores):
            raise ValueError("Evaluation scores must be between 0 and 3")

        if len(scores) != len(results):
            print(f"Warning: Evaluator returned {len(scores)} scores for {len(results)} docs.")
            while len(scores) < len(results):
                scores.append(0)

        return scores[:len(results)]
    except Exception as e:
        print(f"Warning: Failed to process evaluator response ({e}). Defaulting to 0s.")
        return [0] * len(results)

def _generate_answer(query: str, results: list[dict], template: str, label: str) -> str:
    context = _format_results(results)
    prompt = template.format(query=query, doc_list_str=context, results=context)
    try:
        return invoke_llm(prompt)["response"].strip()
    except Exception as e:
        print(f"Warning: LLM invocation failed for {label} ({e}). Defaulting to empty string.")
        return ""


def augmented_generator(query: str, results: list[dict]) -> str:
    return _generate_answer(query, results, AUGMENTED_GENERATION, "augmented generator")


def summarizer(query: str, results: list[dict]) -> str:
    return _generate_answer(query, results, SUMMARIZER, "summarizer")


def citations_generator(query: str, results: list[dict]) -> str:
    return _generate_answer(query, results, CITATIONS, "citations generator")


def question_handler(query: str, results: list[dict]) -> str:
    return _generate_answer(query, results, QUESTION, "question handler")

def image_describer(mime: str, image_path: str, query: str) -> tuple[str, int]:
    client = get_client()

    with open(image_path, "rb") as f:
        image_data = f.read()

    data_url = f"data:{mime};base64,{base64.b64encode(image_data).decode()}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": IMAGE_DESCRIBER.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": query or "Describe this image."}
            ],
        }
    ]

    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )

    return (
        (completion.choices[0].message.content or "").strip(),
        completion.usage.total_tokens if completion.usage else 0,
    )
    