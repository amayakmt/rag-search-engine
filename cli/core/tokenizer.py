import string
from nltk.stem import PorterStemmer
from config import STOPWORDS

PUNCTUATION_TRANSLATOR = str.maketrans("", "", string.punctuation)
STEMMER = PorterStemmer()

def _clean_token(text: str) -> str:
    return text.lower().translate(PUNCTUATION_TRANSLATOR)

def load_stopwords() -> set[str]:
    with open(STOPWORDS, "r", encoding="utf-8") as f:
        return {_clean_token(word) for word in f.read().splitlines() if word.strip()}

CACHED_STOPWORDS: set[str] = load_stopwords()

def _remove_stopwords_and_stem(tokens: list[str]) -> list[str]:
    return [
        STEMMER.stem(token)
        for token in tokens
        if token not in CACHED_STOPWORDS
    ]

def tokenize_text(text: str) -> list[str]:
    cleaned = _clean_token(text)
    return _remove_stopwords_and_stem(cleaned.split())

def tokenize_term(term: str) -> str:
    tokenized_term = tokenize_text(term)
    if len(tokenized_term) != 1:
        raise ValueError(f"A term must be a single word without spaces. Input: {term}")

    return tokenized_term[0]