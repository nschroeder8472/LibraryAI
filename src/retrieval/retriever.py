"""Retrieval component."""
from typing import List, Dict
from ..embeddings.embedder import Embedder
from ..embeddings.vector_store import VectorStore
import logging

logger = logging.getLogger(__name__)

class Retriever:
    """Retrieve relevant chunks for a query."""

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

    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User query text

        Returns:
            List of relevant chunks with metadata and scores
        """
        # Embed query
        logger.info(f"Embedding query: {query[:50]}...")
        query_embedding = self.embedder.embed_text(query)

        # Search vector store
        results = self.vector_store.search(query_embedding, self.top_k)

        # Filter by threshold (higher cosine similarity = more similar)
        # Score range is -1 to 1 for cosine similarity
        # A threshold of 0 disables filtering
        if self.similarity_threshold > 0:
            filtered_results = [
                r for r in results
                if r["similarity_score"] >= self.similarity_threshold
            ]
        else:
            filtered_results = results

        logger.info(f"Retrieved {len(filtered_results)} chunks above threshold")

        return filtered_results

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
            context_part = f"""
Context {idx}:
Book: "{chunk['book_title']}" by {chunk['book_author']}
Chapter: {chunk['chapter_title']}
Text: {chunk['text']}
---"""
            context_parts.append(context_part)

        return "\n".join(context_parts)
