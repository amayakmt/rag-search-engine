import re

def chunk_text_by_words(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("Overlap must be strictly less than chunk_size")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), step)
    ]

def chunk_by_text_sentences(text: str, max_chunk_size: int, overlap: int) -> list[str]:
    if overlap >= max_chunk_size:
        raise ValueError("Overlap must be strictly less than chunk_size")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")

    stripped_text = text.strip()
    if not stripped_text:
        return []

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", stripped_text) if s.strip()]
    if not sentences:
        return []

    step = max_chunk_size - overlap
    chunks = []
    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i : i + max_chunk_size]
        if i > 0 and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))

    return chunks