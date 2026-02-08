"""
Configuration management for LibraryAI RAG system.

This module provides centralized configuration using dataclasses for all
components of the RAG pipeline.

All settings can be overridden via environment variables for Docker/deployment use.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch


def _detect_device() -> str:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _env_bool(key: str, default: bool = False) -> bool:
    """Read a boolean from an environment variable."""
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes")


@dataclass
class DataConfig:
    """Configuration for data directories and file paths."""

    # Base directories
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"

    # Data subdirectories
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    vector_store_dir: Path = data_dir / "vector_store"

    # Index files
    vector_index_path: Path = vector_store_dir / "faiss_index.bin"
    metadata_path: Path = vector_store_dir / "metadata.json"

    def __post_init__(self):
        """Apply env var overrides and ensure all directories exist."""
        if os.environ.get("DATA_DIR"):
            self.data_dir = Path(os.environ["DATA_DIR"])

        self.raw_dir = Path(os.environ.get("RAW_DIR", str(self.data_dir / "raw")))
        self.processed_dir = Path(os.environ.get("PROCESSED_DIR", str(self.data_dir / "processed")))
        self.vector_store_dir = Path(os.environ.get("VECTOR_STORE_DIR", str(self.data_dir / "vector_store")))

        # Recompute derived paths
        self.vector_index_path = self.vector_store_dir / "faiss_index.bin"
        self.metadata_path = self.vector_store_dir / "metadata.json"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""

    # Chunk size in characters
    chunk_size: int = 1000

    # Overlap between chunks in characters
    chunk_overlap: int = 200

    # Text separators for recursive splitting (in order of priority)
    separators: List[str] = None

    # Whether to keep separator in chunks
    keep_separator: bool = True

    def __post_init__(self):
        """Apply env var overrides and set default separators."""
        if os.environ.get("CHUNK_SIZE"):
            self.chunk_size = int(os.environ["CHUNK_SIZE"])
        if os.environ.get("CHUNK_OVERLAP"):
            self.chunk_overlap = int(os.environ["CHUNK_OVERLAP"])

        if self.separators is None:
            self.separators = [
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence breaks
                ", ",    # Clause breaks
                " ",     # Word breaks
                ""       # Character breaks (fallback)
            ]


@dataclass
class EmbeddingConfig:
    """Configuration for text embeddings."""

    # Sentence transformer model name
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Device for inference ('cuda', 'mps', or 'cpu')
    device: str = "cpu"

    # Batch size for embedding generation
    batch_size: int = 32

    # Normalize embeddings to unit length
    normalize_embeddings: bool = True

    # Expected embedding dimension (384 for all-MiniLM-L6-v2)
    embedding_dim: int = 384

    def __post_init__(self):
        """Apply env var overrides."""
        if os.environ.get("EMBEDDING_MODEL"):
            self.model_name = os.environ["EMBEDDING_MODEL"]
        if os.environ.get("EMBEDDING_BATCH_SIZE"):
            self.batch_size = int(os.environ["EMBEDDING_BATCH_SIZE"])

        if os.environ.get("EMBEDDING_DEVICE"):
            self.device = os.environ["EMBEDDING_DEVICE"]
        elif _env_bool("AUTO_DETECT_DEVICE", default=True):
            self.device = _detect_device()


@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""

    # Number of top results to retrieve
    top_k: int = 5

    # Minimum similarity score threshold (0-1, higher is more similar)
    similarity_threshold: float = 0.0

    # FAISS search parameters
    # For L2 distance: smaller values = more similar
    # We use L2 distance by default
    distance_metric: str = "l2"

    def __post_init__(self):
        """Apply env var overrides."""
        if os.environ.get("RETRIEVAL_TOP_K"):
            self.top_k = int(os.environ["RETRIEVAL_TOP_K"])
        if os.environ.get("SIMILARITY_THRESHOLD"):
            self.similarity_threshold = float(os.environ["SIMILARITY_THRESHOLD"])


@dataclass
class GenerationConfig:
    """Configuration for language model generation."""

    # Model identifier
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Device for inference
    device: str = "cpu"

    # Generation parameters
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True

    # Whether to use 8-bit quantization (requires bitsandbytes)
    use_8bit: bool = False

    # Whether to use 4-bit quantization (requires bitsandbytes)
    use_4bit: bool = False

    def __post_init__(self):
        """Apply env var overrides."""
        if os.environ.get("GENERATION_MODEL"):
            self.model_name = os.environ["GENERATION_MODEL"]
        if os.environ.get("MAX_NEW_TOKENS"):
            self.max_new_tokens = int(os.environ["MAX_NEW_TOKENS"])
        if os.environ.get("TEMPERATURE"):
            self.temperature = float(os.environ["TEMPERATURE"])
        if os.environ.get("TOP_P"):
            self.top_p = float(os.environ["TOP_P"])
        if os.environ.get("TOP_K"):
            self.top_k = int(os.environ["TOP_K"])
        if os.environ.get("USE_8BIT"):
            self.use_8bit = _env_bool("USE_8BIT")
        if os.environ.get("USE_4BIT"):
            self.use_4bit = _env_bool("USE_4BIT")

        if os.environ.get("GENERATION_DEVICE"):
            self.device = os.environ["GENERATION_DEVICE"]
        elif _env_bool("AUTO_DETECT_DEVICE", default=True):
            self.device = _detect_device()


@dataclass
class Config:
    """Global configuration object combining all sub-configurations."""

    data: DataConfig = None
    chunking: ChunkingConfig = None
    embedding: EmbeddingConfig = None
    retrieval: RetrievalConfig = None
    generation: GenerationConfig = None

    def __post_init__(self):
        """Initialize sub-configurations if not provided."""
        if self.data is None:
            self.data = DataConfig()
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.retrieval is None:
            self.retrieval = RetrievalConfig()
        if self.generation is None:
            self.generation = GenerationConfig()

    def __repr__(self) -> str:
        """Return a formatted string representation of the configuration."""
        lines = ["LibraryAI Configuration:"]
        lines.append("\nData Configuration:")
        lines.append(f"  Raw directory: {self.data.raw_dir}")
        lines.append(f"  Processed directory: {self.data.processed_dir}")
        lines.append(f"  Vector store directory: {self.data.vector_store_dir}")

        lines.append("\nChunking Configuration:")
        lines.append(f"  Chunk size: {self.chunking.chunk_size}")
        lines.append(f"  Chunk overlap: {self.chunking.chunk_overlap}")

        lines.append("\nEmbedding Configuration:")
        lines.append(f"  Model: {self.embedding.model_name}")
        lines.append(f"  Device: {self.embedding.device}")
        lines.append(f"  Batch size: {self.embedding.batch_size}")

        lines.append("\nRetrieval Configuration:")
        lines.append(f"  Top K: {self.retrieval.top_k}")
        lines.append(f"  Similarity threshold: {self.retrieval.similarity_threshold}")

        lines.append("\nGeneration Configuration:")
        lines.append(f"  Model: {self.generation.model_name}")
        lines.append(f"  Device: {self.generation.device}")
        lines.append(f"  Max new tokens: {self.generation.max_new_tokens}")

        return "\n".join(lines)


# Global configuration instance
config = Config()
