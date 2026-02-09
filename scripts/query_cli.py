#!/usr/bin/env python3
"""
Query the indexed library.

This script loads the vector store and answers questions about your library.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.query import QueryPipeline
from src.embeddings.vector_store import VectorStore
from src.config import config
from src.utils.logging_config import setup_logging


def main():
    """Main entry point for querying."""
    parser = argparse.ArgumentParser(
        description="Query your indexed book library"
    )
    parser.add_argument(
        "query",
        type=str,
        help="Your question or query"
    )
    parser.add_argument(
        "--type",
        default="qa",
        choices=["qa", "recommendation", "passage_location"],
        help="Type of query (default: qa)"
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING for cleaner output)"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)

    # Check if index exists
    chroma_dir = config.data.vector_store_dir / "chroma.sqlite3"
    if not chroma_dir.exists():
        print("Error: Vector index not found.")
        print(f"Please build the index first using:")
        print(f"  python scripts/build_index.py")
        print(f"or")
        print(f"  python main.py index --library-dir data/raw")
        sys.exit(1)

    try:
        # Load vector store
        print("Loading vector store...")
        vector_store = VectorStore.load(config.data.vector_store_dir)
        stats = vector_store.get_stats()
        print(f"Loaded index with {stats['total_vectors']} vectors")
        print()

        # Create query pipeline
        print("Initializing query pipeline...")
        pipeline = QueryPipeline(vector_store)
        print()

        # Process query
        print(f"Query: {args.query}")
        print()
        result = pipeline.query(args.query, query_type=args.type)

        # Display results
        print("=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(result["answer"])
        print()

        # Display sources
        if result["contexts"]:
            print("=" * 60)
            print(f"SOURCES ({len(result['contexts'])} contexts retrieved):")
            print("=" * 60)
            for i, ctx in enumerate(result["contexts"], 1):
                print(f"\n{i}. \"{ctx['book_title']}\" by {ctx['book_author']}")
                print(f"   Chapter: {ctx['chapter_title']}")
                print(f"   Similarity score: {ctx['similarity_score']:.4f}")
                print(f"   Text preview: {ctx['text'][:150]}...")

    except Exception as e:
        logger.error(f"Error during query: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
