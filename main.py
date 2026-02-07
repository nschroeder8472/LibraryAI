#!/usr/bin/env python3
"""
LibraryAI - RAG-based Q&A system for personal ebook libraries.

This is the main CLI interface for LibraryAI. It supports:
- Indexing EPUB files into a vector store
- Querying the indexed library
- Interactive query mode
"""

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline.indexing import IndexingPipeline
from src.pipeline.query import QueryPipeline
from src.embeddings.vector_store import VectorStore
from src.config import config
from src.utils.logging_config import setup_logging


def index_command(args):
    """Handle the index command."""
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("LibraryAI Indexing")
    logger.info("=" * 60)

    # Set library directory
    library_dir = Path(args.library_dir) if args.library_dir else config.data.raw_dir

    # Verify directory exists
    if not library_dir.exists():
        logger.error(f"Library directory does not exist: {library_dir}")
        logger.info(f"Please create the directory and add EPUB files to it.")
        return 1

    # Check for EPUB files
    epub_files = list(library_dir.glob("*.epub"))
    if not epub_files:
        logger.error(f"No EPUB files found in {library_dir}")
        logger.info("Please add EPUB files to the directory and try again.")
        return 1

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
        logger.info("  python main.py query \"your question here\"")
        logger.info("or")
        logger.info("  python main.py interactive")

        return 0

    except Exception as e:
        logger.error(f"Error during indexing: {e}", exc_info=True)
        return 1


def query_command(args):
    """Handle the query command."""
    logger = logging.getLogger(__name__)

    # Check if index exists
    index_path = config.data.vector_store_dir / "faiss_index.bin"
    if not index_path.exists():
        print("Error: Vector index not found.")
        print(f"Please build the index first using:")
        print(f"  python main.py index --library-dir data/raw")
        return 1

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

        # Display sources if requested
        if args.show_sources and result["contexts"]:
            print("=" * 60)
            print(f"SOURCES ({len(result['contexts'])} contexts retrieved):")
            print("=" * 60)
            for i, ctx in enumerate(result["contexts"], 1):
                print(f"\n{i}. \"{ctx['book_title']}\" by {ctx['book_author']}")
                print(f"   Chapter: {ctx['chapter_title']}")
                print(f"   Similarity score: {ctx['similarity_score']:.4f}")
                print(f"   Text preview: {ctx['text'][:150]}...")

        return 0

    except Exception as e:
        logger.error(f"Error during query: {e}", exc_info=True)
        return 1


def interactive_command(args):
    """Handle the interactive command."""
    logger = logging.getLogger(__name__)

    # Check if index exists
    index_path = config.data.vector_store_dir / "faiss_index.bin"
    if not index_path.exists():
        print("Error: Vector index not found.")
        print(f"Please build the index first using:")
        print(f"  python main.py index --library-dir data/raw")
        return 1

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

        print("=" * 60)
        print("LibraryAI Interactive Mode")
        print("=" * 60)
        print("Enter your questions (type 'quit' or 'exit' to stop)")
        print("Commands:")
        print("  /type qa - Switch to Q&A mode (default)")
        print("  /type recommendation - Switch to recommendation mode")
        print("  /type passage_location - Switch to passage location mode")
        print("  /sources - Toggle source display")
        print("=" * 60)
        print()

        query_type = "qa"
        show_sources = False

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break

                # Handle commands
                if user_input.startswith('/'):
                    if user_input.startswith('/type '):
                        new_type = user_input[6:].strip()
                        if new_type in ['qa', 'recommendation', 'passage_location']:
                            query_type = new_type
                            print(f"Switched to {query_type} mode")
                        else:
                            print(f"Unknown type: {new_type}")
                    elif user_input == '/sources':
                        show_sources = not show_sources
                        print(f"Source display: {'ON' if show_sources else 'OFF'}")
                    else:
                        print(f"Unknown command: {user_input}")
                    continue

                # Process query
                print()
                result = pipeline.query(user_input, query_type=query_type)

                # Display answer
                print("Assistant:", result["answer"])
                print()

                # Display sources if enabled
                if show_sources and result["contexts"]:
                    print(f"[{len(result['contexts'])} sources used]")
                    for i, ctx in enumerate(result["contexts"], 1):
                        print(f"  {i}. \"{ctx['book_title']}\" - {ctx['chapter_title']}")
                    print()

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break

        return 0

    except Exception as e:
        logger.error(f"Error in interactive mode: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LibraryAI - RAG-based Q&A for your ebook library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index your library
  python main.py index --library-dir data/raw

  # Query your library
  python main.py query "What books discuss AI?"

  # Interactive mode
  python main.py interactive
        """
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Index command
    index_parser = subparsers.add_parser(
        "index",
        help="Build vector index from EPUB library"
    )
    index_parser.add_argument(
        "--library-dir",
        type=str,
        default=None,
        help=f"Directory containing EPUB files (default: {config.data.raw_dir})"
    )

    # Query command
    query_parser = subparsers.add_parser(
        "query",
        help="Query the indexed library"
    )
    query_parser.add_argument(
        "query",
        type=str,
        help="Your question or query"
    )
    query_parser.add_argument(
        "--type",
        default="qa",
        choices=["qa", "recommendation", "passage_location"],
        help="Type of query (default: qa)"
    )
    query_parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Display source contexts"
    )

    # Interactive command
    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Enter interactive query mode"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)

    # Use WARNING for query command to keep output clean
    if args.command == "query":
        setup_logging(level=logging.WARNING)
    else:
        setup_logging(level=log_level)

    # Route to appropriate command
    if args.command == "index":
        return index_command(args)
    elif args.command == "query":
        return query_command(args)
    elif args.command == "interactive":
        return interactive_command(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
