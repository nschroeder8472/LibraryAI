"""Vector store using FAISS."""
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """FAISS-based vector store for chunk embeddings."""

    def __init__(self, embedding_dim: int):
        """
        Initialize vector store.

        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
        self.chunks_metadata = []  # Store chunk metadata
        self.is_trained = False

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Dict]):
        """
        Add embeddings and metadata to the index.

        Args:
            embeddings: Array of embeddings (N, embedding_dim)
            chunks: List of chunk dictionaries with metadata
        """
        # Ensure correct shape
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))

        # Store metadata
        self.chunks_metadata.extend(chunks)

        logger.info(f"Added {len(embeddings)} embeddings. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of chunk dictionaries with similarity scores
        """
        # Ensure correct shape
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search FAISS index
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            top_k
        )

        # Retrieve metadata and add scores
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks_metadata):
                chunk = self.chunks_metadata[idx].copy()
                # Convert L2 distance to similarity score (lower is better)
                chunk["similarity_score"] = float(dist)
                results.append(chunk)

        return results

    def save(self, save_dir: Path):
        """
        Save index and metadata to disk.

        Args:
            save_dir: Directory to save to
        """
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = save_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))

        # Save metadata
        metadata_path = save_dir / "chunks_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.chunks_metadata, f)

        logger.info(f"Saved vector store to {save_dir}")

    @classmethod
    def load(cls, load_dir: Path) -> 'VectorStore':
        """
        Load index and metadata from disk.

        Args:
            load_dir: Directory to load from

        Returns:
            VectorStore instance
        """
        # Load FAISS index
        index_path = load_dir / "faiss_index.bin"
        index = faiss.read_index(str(index_path))

        # Load metadata
        metadata_path = load_dir / "chunks_metadata.pkl"
        with open(metadata_path, 'rb') as f:
            chunks_metadata = pickle.load(f)

        # Create instance
        embedding_dim = index.d
        vector_store = cls(embedding_dim)
        vector_store.index = index
        vector_store.chunks_metadata = chunks_metadata
        vector_store.is_trained = True

        logger.info(f"Loaded vector store from {load_dir} with {index.ntotal} vectors")

        return vector_store

    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "total_chunks": len(self.chunks_metadata)
        }
