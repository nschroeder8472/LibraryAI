"""Indexing pipeline orchestration."""
from pathlib import Path
from typing import List, Dict
import json
import logging

from ..data.epub_parser import EPUBParser
from ..data.chunker import TextChunker
from ..data.series_manager import SeriesManager
from ..embeddings.embedder import Embedder
from ..embeddings.vector_store import VectorStore
from ..config import config

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """Orchestrate the indexing of book library."""

    def __init__(self):
        """Initialize pipeline components."""
        self.parser = EPUBParser()
        self.chunker = TextChunker(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            separators=config.chunking.separators,
            parent_chunk_size=config.chunking.parent_chunk_size,
            parent_chunk_overlap=config.chunking.parent_chunk_overlap,
            use_hierarchical=config.chunking.use_hierarchical,
        )
        self.embedder = Embedder(
            model_name=config.embedding.model_name,
            device=config.embedding.device,
            batch_size=config.embedding.batch_size
        )
        self.series_manager = SeriesManager(
            config_path=config.data.series_config_path
        )
        # Use persistent ChromaDB directory directly
        self.vector_store = VectorStore(
            self.embedder.embedding_dim,
            persist_dir=config.data.vector_store_dir,
        )

    def index_library(self, library_dir: Path = None) -> VectorStore:
        """
        Index all books in library directory.

        Args:
            library_dir: Directory containing EPUB files

        Returns:
            Populated vector store
        """
        if library_dir is None:
            library_dir = config.data.raw_dir

        logger.info(f"Starting indexing pipeline for {library_dir}")

        # Step 1: Parse EPUBs
        logger.info("Step 1/5: Parsing EPUB files...")
        library_data = self.parser.parse_library(library_dir)
        logger.info(f"Parsed {len(library_data)} books")

        # Save parsed data
        parsed_path = config.data.processed_dir / "parsed_books.json"
        with open(parsed_path, 'w') as f:
            json_safe_data = [{k: v for k, v in book.items() if k != 'embedding'}
                             for book in library_data]
            json.dump(json_safe_data, f, indent=2)
        logger.info(f"Saved parsed data to {parsed_path}")

        # Step 2: Detect series relationships
        logger.info("Step 2/5: Detecting series relationships...")
        self.series_manager.detect_series_from_books(library_data)
        self.series_manager.save_config(config.data.series_config_path)
        logger.info(f"Detected {len(self.series_manager.series)} series")

        # Step 3: Chunk text with series-aware metadata
        logger.info("Step 3/5: Chunking text...")
        all_chunks = self.chunker.chunk_library(
            library_data, series_manager=self.series_manager
        )
        logger.info(f"Created {len(all_chunks)} chunks")

        # Step 4: Generate embeddings (embed child text, not parent)
        logger.info("Step 4/5: Generating embeddings...")
        embeddings, chunks_with_embeddings = self.embedder.embed_chunks(all_chunks)

        # Step 5: Build vector store
        logger.info("Step 5/5: Building vector store...")
        self.vector_store.add_embeddings(embeddings, chunks_with_embeddings)
        self.vector_store.save(config.data.vector_store_dir)

        logger.info("Indexing pipeline complete!")
        logger.info(f"Stats: {self.vector_store.get_stats()}")

        return self.vector_store
