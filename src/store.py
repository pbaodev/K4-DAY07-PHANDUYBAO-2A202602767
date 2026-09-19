from __future__ import annotations

from typing import Any, Callable

from .chunking import compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    In-memory store: each record keeps the chunk text, a copy of its metadata and its embedding.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        # In-memory only. No test needs ChromaDB, and the old try-block set _use_chroma = True
        # before any client existed, so a machine with chromadb installed would route every
        # method into an unimplemented branch.
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # Copy so later changes to the caller's dict never leak into the store.
        metadata = dict(doc.metadata)
        # delete_document() keys on doc_id. A chunk ("file#3") carries its source file's doc_id
        # from the caller; a whole Document falls back to its own id.
        metadata.setdefault("doc_id", doc.id)
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content),
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not records or top_k <= 0:
            return []
        query_embedding = self._embedding_fn(query)
        # Full cosine rather than bare dot product: identical for normalized vectors (mock, local),
        # still correct for a backend that returns unnormalized ones.
        scored = [
            {
                "id": record["id"],
                "content": record["content"],
                "metadata": dict(record["metadata"]),
                "score": compute_similarity(query_embedding, record["embedding"]),
            }
            for record in records
        ]
        scored.sort(key=lambda result: result["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it as one record.

        No chunking happens here: one Document in, one record out. Chunk before calling.
        """
        for doc in docs:
            self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query, sorted by cosine score descending.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        A filter value that is a list/tuple/set matches any of its items,
        e.g. {"audience": ["student", "all"]}.
        """
        if not metadata_filter:
            return self._search_records(query, self._store, top_k)
        # Filter BEFORE ranking: filtering the top_k afterwards can leave zero results
        # even though matching chunks exist further down the ranking.
        candidates = [record for record in self._store if _matches(record["metadata"], metadata_filter)]
        return self._search_records(query, candidates, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        size_before = len(self._store)
        self._store = [record for record in self._store if record["metadata"].get("doc_id") != doc_id]
        return len(self._store) < size_before


def _matches(metadata: dict, metadata_filter: dict) -> bool:
    for key, expected in metadata_filter.items():
        value = metadata.get(key)
        if isinstance(expected, (list, tuple, set)):
            if value not in expected:
                return False
        elif value != expected:
            return False
    return True
