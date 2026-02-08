"""Cross-encoder reranker for improving retrieval precision."""
from typing import List, Dict
from sentence_transformers import CrossEncoder
import logging

logger = logging.getLogger(__name__)


class Reranker:
    """Rerank retrieved chunks using a cross-encoder model.

    Cross-encoders score query-document pairs jointly, producing much more
    accurate relevance scores than bi-encoder (embedding) similarity alone.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: str = "cpu", top_n: int = 5):
        """
        Initialize reranker.

        Args:
            model_name: Cross-encoder model name
            device: Device to use
            top_n: Number of top results to keep after reranking
        """
        self.top_n = top_n
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name, device=device)
        logger.info("Reranker model loaded")

    def rerank(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """
        Rerank chunks by cross-encoder relevance to the query.

        Args:
            query: User query
            chunks: Retrieved chunks with metadata

        Returns:
            Reranked and filtered list of chunks (top_n most relevant)
        """
        if not chunks:
            return chunks

        # Build query-document pairs for the cross-encoder
        pairs = [[query, chunk["text"]] for chunk in chunks]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach reranker scores and sort by descending relevance
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)

        # Keep only top_n
        reranked = reranked[:self.top_n]

        logger.info(
            f"Reranked {len(chunks)} chunks → kept top {len(reranked)} "
            f"(scores: {reranked[0]['rerank_score']:.3f} to {reranked[-1]['rerank_score']:.3f})"
        )

        return reranked
