from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        # Lookbehind splits on the whitespace *after* . ! ? so the punctuation stays with its sentence.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        n = self.max_sentences_per_chunk
        return [" ".join(sentences[i : i + n]) for i in range(0, len(sentences), n)]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return [c.strip() for c in self._split(text, self.separators) if c.strip()]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case 1: already fits.
        if len(current_text.strip()) <= self.chunk_size:
            return [current_text]

        # Base case 2: no separator left (or "" = split anywhere) -> hard cut by chunk_size.
        if not remaining_separators or remaining_separators[0] == "":
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        sep, rest = remaining_separators[0], remaining_separators[1:]

        # Base case 3: this separator does not occur -> try the next, finer one.
        if sep not in current_text:
            return self._split(current_text, rest)

        # Keep the separator attached to each piece so re-joining is lossless.
        parts = current_text.split(sep)
        pieces = [p + sep for p in parts[:-1]] + [parts[-1]]

        chunks: list[str] = []
        buffer = ""
        for piece in pieces:
            # Recurse down: a piece that is still too long gets split with finer separators.
            # The pending buffer goes in with it so a short heading sticks to its first sub-chunk,
            # and the last sub-chunk stays open so it can still merge with what follows.
            if len(piece.strip()) > self.chunk_size:
                sub_chunks = self._split(buffer + piece, rest)
                chunks.extend(sub_chunks[:-1])
                buffer = sub_chunks[-1] if sub_chunks else ""
                continue
            # Merge up: pack adjacent small pieces until the next one would overflow.
            if len((buffer + piece).strip()) <= self.chunk_size:
                buffer += piece
            else:
                if buffer.strip():
                    chunks.append(buffer)
                buffer = piece
        if buffer.strip():
            chunks.append(buffer)
        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # overlap=0 so every strategy produces non-overlapping chunks and counts are comparable.
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=0),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }
        result: dict = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            lengths = [len(c) for c in chunks]
            count = len(chunks)
            result[name] = {
                "count": count,
                "avg_length": round(sum(lengths) / count, 2) if count else 0.0,
                "min_length": min(lengths) if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
                "chunks": chunks,
            }
        return result
