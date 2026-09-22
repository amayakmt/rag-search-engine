import os
from typing import TypedDict
from dotenv import load_dotenv
from openai import OpenAI

from config import MODEL, LLM_BASE_URL
from core.llm_prompts import SPELL_CHECKER, REWRITER

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

def spell_checker(query: str) -> LLMResponse:
    return invoke_llm(f"{SPELL_CHECKER} '{query}'")

def rewriter(query: str) -> LLMResponse:
    return invoke_llm(f"{REWRITER} '{query}'")