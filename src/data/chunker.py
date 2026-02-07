"""Text chunking utilities."""
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

class TextChunker:
    """Chunk text documents into smaller passages."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50,
                 separators: List[str] = None):
        """
        Initialize chunker.

        Args:
            chunk_size: Target size of each chunk (in characters)
            chunk_overlap: Overlap between consecutive chunks
            separators: List of separators to use for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        return self.splitter.split_text(text)

    def chunk_chapter(self, chapter: Dict, book_metadata: Dict) -> List[Dict]:
        """
        Chunk a single chapter and attach metadata.

        Args:
            chapter: Chapter dictionary with text and metadata
            book_metadata: Book-level metadata

        Returns:
            List of chunk dictionaries with metadata
        """
        text = chapter["text"]
        chunks_text = self.chunk_text(text)

        chunks = []
        for idx, chunk_text in enumerate(chunks_text):
            chunk = {
                "text": chunk_text,
                "book_title": book_metadata["title"],
                "book_author": book_metadata["author"],
                "chapter_title": chapter["title"],
                "chapter_order": chapter["order"],
                "chunk_index": idx,
                "total_chunks_in_chapter": len(chunks_text)
            }
            chunks.append(chunk)

        return chunks

    def chunk_book(self, book_data: Dict) -> List[Dict]:
        """
        Chunk all chapters in a book.

        Args:
            book_data: Book dictionary from parser

        Returns:
            List of all chunks with metadata
        """
        all_chunks = []

        book_metadata = {
            "title": book_data["title"],
            "author": book_data["author"]
        }

        for chapter in book_data["chapters"]:
            chapter_chunks = self.chunk_chapter(chapter, book_metadata)
            all_chunks.extend(chapter_chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {book_data['title']}")

        return all_chunks

    def chunk_library(self, library_data: List[Dict]) -> List[Dict]:
        """
        Chunk all books in a library.

        Args:
            library_data: List of book dictionaries

        Returns:
            List of all chunks from all books
        """
        all_chunks = []

        for book_data in library_data:
            book_chunks = self.chunk_book(book_data)
            all_chunks.extend(book_chunks)

        logger.info(f"Total chunks created: {len(all_chunks)}")

        return all_chunks
