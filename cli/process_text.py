# helper functions to preprocess text for searching
import string
from pathlib import Path
from nltk.stem import PorterStemmer


BASE_DIR = Path(__file__).resolve().parent.parent
STOPWORDS = BASE_DIR / "data" / "stopwords.txt"


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


def process_text(text: str) -> list[str]:
    lowered = text.lower()
    mask = str.maketrans("", "", string.punctuation)
    no_punctuation = lowered.translate(mask)
    tokenized = no_punctuation.split()
    no_stopwords = _remove_stopwords_and_stem(tokenized)
    

    return no_stopwords
