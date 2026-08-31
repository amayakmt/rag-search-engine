# helper functions to preprocess text for searching
import string
from pathlib import Path
from nltk.stem import PorterStemmer
from config import STOPWORDS

def _remove_stopwords_and_stem(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    result = []

    with open(STOPWORDS, "r", encoding="utf-8") as f:
        stopwords = set(f.read().splitlines())

        for token in tokens:
            if token in stopwords:
                continue
            result.append(stemmer.stem(token))

        return result


def tokenize_text(text: str) -> list[str]:
    lowered = text.lower()
    mask = str.maketrans("", "", string.punctuation)
    no_punctuation = lowered.translate(mask)
    tokenized = no_punctuation.split()
    no_stopwords = _remove_stopwords_and_stem(tokenized)
    
    return no_stopwords

def tokenize_term(term: str) -> str:
    tokenized_term = tokenize_text(term)
    if len(tokenized_term) != 1:
        raise ValueError(f"A term must be a single word without spaces. Input:{term}")

    return tokenized_term[0]