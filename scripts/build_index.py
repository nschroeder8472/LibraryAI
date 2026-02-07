#!/usr/bin/env python3
"""
Build vector index from EPUB library.

This script parses EPUB files, chunks them, generates embeddings,
and builds a FAISS vector store for retrieval.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.indexing import IndexingPipeline
from src.config import config
from src.utils.logging_config import setup_logging


def main():
    """Main entry point for index building."""
    parser = argparse.ArgumentParser(
        description="Build vector index from EPUB library"
    )
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=None,
        help=f"Directory containing EPUB files (default: {config.data.raw_dir})"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("LibraryAI Index Builder")
    logger.info("=" * 60)

    # Set library directory
    library_dir = args.library_dir if args.library_dir else config.data.raw_dir

    # Verify directory exists
    if not library_dir.exists():
        logger.error(f"Library directory does not exist: {library_dir}")
        logger.info(f"Please create the directory and add EPUB files to it.")
        sys.exit(1)

    # Check for EPUB files
    epub_files = list(library_dir.glob("*.epub"))
    if not epub_files:
        logger.error(f"No EPUB files found in {library_dir}")
        logger.info("Please add EPUB files to the directory and try again.")
        sys.exit(1)

    logger.info(f"Found {len(epub_files)} EPUB files in {library_dir}")
    logger.info("")

    # Create and run pipeline
    try:
        pipeline = IndexingPipeline()
        vector_store = pipeline.index_library(library_dir)

        logger.info("")
        logger.info("=" * 60)
        logger.info("Indexing Complete!")
        logger.info("=" * 60)
        stats = vector_store.get_stats()
        logger.info(f"Total vectors indexed: {stats['total_vectors']}")
        logger.info(f"Total chunks: {stats['total_chunks']}")
        logger.info(f"Embedding dimension: {stats['embedding_dim']}")
        logger.info(f"Index saved to: {config.data.vector_store_dir}")
        logger.info("")
        logger.info("You can now query your library using:")
        logger.info("  python scripts/query_cli.py \"your question here\"")
        logger.info("or")
        logger.info("  python main.py query \"your question here\"")

    except Exception as e:
        logger.error(f"Error during indexing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
