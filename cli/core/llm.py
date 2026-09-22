import os
import time
from typing import TypedDict
from dotenv import load_dotenv
from openai import OpenAI

from config import MODEL, LLM_BASE_URL
from core.llm_prompts import SPELL_CHECKER, REWRITER, EXPANSION, INDIVIDUAL_RERANK

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
        "response": str(completion.choices[0].message.content),
        "prompt_tokens": completion.usage.prompt_tokens,
        "response_tokens": completion.usage.completion_tokens
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
            document=doc.get("document", "")
        )
        raw_score = invoke_llm(prompt)["response"]

        try:
            doc["individual_rerank_score"] = float(raw_score.strip())
        except ValueError:
            doc["individual_rerank_score"] = 0.0

        time.sleep(3)

    return sorted(documents, key=lambda item: item["individual_rerank_score"], reverse=True)