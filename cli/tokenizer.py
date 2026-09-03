# helper functions to preprocess text for searching
import string
from pathlib import Path
from nltk.stem import PorterStemmer
from config import STOPWORDS

def _clean_token(text: str) -> str:
    lowered = text.lower()
    mask = str.maketrans("", "", string.punctuation)
    return lowered.translate(mask)


def load_stopwords() -> set[str]:
    with open(STOPWORDS, "r", encoding="utf-8") as f:
        # Preprocess each stop word (strip punctuation and lower) so tokens match
        return {_clean_token(word) for word in f.read().splitlines() if word.strip()}


# Cache stopwords set in memory once instead of reading disk every tokenization
CACHED_STOPWORDS: set[str] = load_stopwords()


def _remove_stopwords_and_stem(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    result = []

    for token in tokens:
        if token in CACHED_STOPWORDS:
            continue
        result.append(stemmer.stem(token))

    return result


def tokenize_text(text: str) -> list[str]:
    cleaned = _clean_token(text)
    tokenized = cleaned.split()
    return _remove_stopwords_and_stem(tokenized)


def tokenize_term(term: str) -> str:
    tokenized_term = tokenize_text(term)
    if len(tokenized_term) != 1:
        raise ValueError(f"A term must be a single word without spaces. Input: {term}")

    return tokenized_term[0]