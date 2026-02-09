"""Retrieval component with scoped and metadata-aware search."""
from typing import List, Dict, Optional
from ..embeddings.embedder import Embedder
from ..embeddings.vector_store import VectorStore
import logging

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieve relevant chunks for a query with optional metadata filtering."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore,
                 top_k: int = 5, similarity_threshold: float = 0.7):
        """
        Initialize retriever.

        Args:
            embedder: Embedder instance
            vector_store: VectorStore instance
            top_k: Number of results to retrieve
            similarity_threshold: Minimum similarity score
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(self, query: str, scope: Optional[Dict] = None) -> List[Dict]:
        """
        Retrieve relevant chunks for a query, optionally scoped to a book or series.

        Args:
            query: User query text
            scope: Optional scope filter, e.g.:
                   {"type": "book", "title": "The Eye of the World"}
                   {"type": "series", "name": "The Wheel of Time"}

        Returns:
            List of relevant chunks with metadata and scores
        """
        logger.info(f"Embedding query: {query[:50]}...")
        query_embedding = self.embedder.embed_text(query)

        # Build ChromaDB where filter from scope
        where = self._build_scope_filter(scope)

        # Search vector store
        results = self.vector_store.search(
            query_embedding, self.top_k, where=where
        )

        # Filter by threshold (higher cosine similarity = more similar)
        if self.similarity_threshold > 0:
            filtered_results = [
                r for r in results
                if r["similarity_score"] >= self.similarity_threshold
            ]
        else:
            filtered_results = results

        # Deduplicate by parent_id to avoid sending the same parent chunk
        # to the LLM multiple times when multiple child chunks match
        filtered_results = self._deduplicate_parents(filtered_results)

        logger.info(f"Retrieved {len(filtered_results)} chunks above threshold")
        return filtered_results

    def _build_scope_filter(self, scope: Optional[Dict]) -> Optional[Dict]:
        """Convert a scope dict to a ChromaDB where filter."""
        if not scope:
            return None

        scope_type = scope.get("type", "")
        if scope_type == "book":
            title = scope.get("title", "")
            if title:
                return {"book_title": title}
        elif scope_type == "series":
            name = scope.get("name", "")
            if name:
                return {"series_name": name}

        return None

    def _deduplicate_parents(self, chunks: List[Dict]) -> List[Dict]:
        """Deduplicate chunks that share the same parent.

        When hierarchical chunking is used, multiple child chunks from the
        same parent may match. We keep only the highest-scoring child per
        parent to avoid redundant context for the LLM.
        """
        seen_parents = set()
        deduplicated = []
        for chunk in chunks:
            parent_id = chunk.get("parent_id", "")
            if parent_id and parent_id in seen_parents:
                continue
            if parent_id:
                seen_parents.add(parent_id)
            deduplicated.append(chunk)
        return deduplicated

    def retrieve_chronological(self, query: str,
                               scope: Optional[Dict] = None) -> List[Dict]:
        """Retrieve chunks and sort them chronologically by global_order.

        Useful for character evolution and timeline questions where the
        order of events matters more than similarity ranking.

        Args:
            query: User query text
            scope: Optional scope filter

        Returns:
            List of relevant chunks sorted by global_order
        """
        chunks = self.retrieve(query, scope=scope)
        chunks.sort(key=lambda c: c.get("global_order", 0))
        return chunks

    def format_context(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into context string for prompt.

        Args:
            chunks: Retrieved chunks

        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant information found in the library."

        context_parts = []
        for idx, chunk in enumerate(chunks, 1):
            series_info = ""
            series_name = chunk.get("series_name", "")
            if series_name:
                book_order = chunk.get("book_order_in_series", 0)
                series_info = f"\nSeries: {series_name} (Book {book_order})"

            context_part = f"""
Context {idx}:
Book: "{chunk['book_title']}" by {chunk['book_author']}{series_info}
Chapter: {chunk['chapter_title']}
Text: {chunk['text']}
---"""
            context_parts.append(context_part)

        return "\n".join(context_parts)
