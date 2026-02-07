"""Embedding generation utilities."""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import torch
import logging

logger = logging.getLogger(__name__)

class Embedder:
    """Generate embeddings for text using sentence transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu", batch_size: int = 32):
        """
        Initialize embedder.

        Args:
            model_name: Name of sentence-transformer model
            device: Device to use (cpu or cuda)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embeddings (num_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    def embed_chunks(self, chunks: List[Dict]) -> tuple[np.ndarray, List[Dict]]:
        """
        Generate embeddings for chunks with metadata.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Tuple of (embeddings array, chunks with metadata)
        """
        texts = [chunk["text"] for chunk in chunks]

        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embed_batch(texts)

        # Attach embedding to each chunk
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        return embeddings, chunks
