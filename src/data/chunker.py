"""Text chunking utilities with hierarchical parent-child strategy."""
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
import uuid

logger = logging.getLogger(__name__)


class TextChunker:
    """Chunk text documents into smaller passages.

    Supports two modes:
    - Flat chunking: Simple fixed-size chunks (original behavior).
    - Hierarchical chunking: Small child chunks for precise retrieval,
      linked to larger parent chunks that provide richer context to the LLM.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50,
                 separators: List[str] = None,
                 parent_chunk_size: int = 1500, parent_chunk_overlap: int = 200,
                 use_hierarchical: bool = True):
        """
        Initialize chunker.

        Args:
            chunk_size: Target size of each child chunk (in characters)
            chunk_overlap: Overlap between consecutive child chunks
            separators: List of separators to use for splitting
            parent_chunk_size: Target size of parent chunks
            parent_chunk_overlap: Overlap between parent chunks
            use_hierarchical: Whether to use hierarchical (parent-child) chunking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.use_hierarchical = use_hierarchical

        if separators is None:
            separators = ["\n\n", "\n", ". ", ", ", " ", ""]

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=separators,
            length_function=len,
        )

        # Keep a flat splitter for non-hierarchical mode (uses child size)
        self.splitter = self.child_splitter

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        return self.splitter.split_text(text)

    def chunk_chapter(self, chapter: Dict, book_metadata: Dict,
                      series_name: Optional[str] = None,
                      book_order: int = 0) -> List[Dict]:
        """
        Chunk a single chapter and attach metadata.

        Uses hierarchical chunking if enabled: creates parent chunks for LLM
        context and child chunks for embedding/retrieval.

        Args:
            chapter: Chapter dictionary with text and metadata
            book_metadata: Book-level metadata
            series_name: Name of the series this book belongs to (if any)
            book_order: Order of this book within its series

        Returns:
            List of chunk dictionaries with metadata
        """
        text = chapter["text"]

        if self.use_hierarchical:
            return self._chunk_chapter_hierarchical(
                text, chapter, book_metadata, series_name, book_order
            )
        return self._chunk_chapter_flat(
            text, chapter, book_metadata, series_name, book_order
        )

    def _chunk_chapter_flat(self, text: str, chapter: Dict,
                            book_metadata: Dict,
                            series_name: Optional[str],
                            book_order: int) -> List[Dict]:
        """Flat chunking: each chunk is both the retrieval unit and the context."""
        chunks_text = self.child_splitter.split_text(text)
        chunks = []
        for idx, chunk_text in enumerate(chunks_text):
            chunk = self._build_chunk_metadata(
                text=chunk_text,
                parent_text=chunk_text,
                parent_id=None,
                chunk_index=idx,
                total_chunks=len(chunks_text),
                chapter=chapter,
                book_metadata=book_metadata,
                series_name=series_name,
                book_order=book_order,
            )
            chunks.append(chunk)
        return chunks

    def _chunk_chapter_hierarchical(self, text: str, chapter: Dict,
                                    book_metadata: Dict,
                                    series_name: Optional[str],
                                    book_order: int) -> List[Dict]:
        """Hierarchical chunking: small child chunks linked to larger parents.

        The child chunk text is used for embedding (more precise retrieval).
        The parent chunk text is what gets sent to the LLM (richer context).
        """
        # Create parent chunks
        parent_texts = self.parent_splitter.split_text(text)
        parents = []
        for p_text in parent_texts:
            parents.append({
                "id": str(uuid.uuid4()),
                "text": p_text,
            })

        # Create child chunks and link each to its parent
        chunks = []
        child_index = 0
        for parent in parents:
            child_texts = self.child_splitter.split_text(parent["text"])
            for c_text in child_texts:
                chunk = self._build_chunk_metadata(
                    text=c_text,
                    parent_text=parent["text"],
                    parent_id=parent["id"],
                    chunk_index=child_index,
                    total_chunks=-1,  # Will be set after all chunks created
                    chapter=chapter,
                    book_metadata=book_metadata,
                    series_name=series_name,
                    book_order=book_order,
                )
                chunks.append(chunk)
                child_index += 1

        # Set total_chunks now that we know the count
        for chunk in chunks:
            chunk["total_chunks_in_chapter"] = len(chunks)

        return chunks

    def _build_chunk_metadata(self, text: str, parent_text: str,
                              parent_id: Optional[str],
                              chunk_index: int, total_chunks: int,
                              chapter: Dict, book_metadata: Dict,
                              series_name: Optional[str],
                              book_order: int) -> Dict:
        """Build a chunk dictionary with all metadata fields."""
        chapter_order = chapter.get("order", 0)
        global_order = book_order * 100000 + chapter_order * 1000 + chunk_index

        return {
            "text": text,
            "parent_text": parent_text,
            "parent_id": parent_id,
            "book_title": book_metadata["title"],
            "book_author": book_metadata["author"],
            "chapter_title": chapter["title"],
            "chapter_order": chapter_order,
            "chunk_index": chunk_index,
            "total_chunks_in_chapter": total_chunks,
            "series_name": series_name or "",
            "book_order_in_series": book_order,
            "global_order": global_order,
        }

    def chunk_book(self, book_data: Dict,
                   series_name: Optional[str] = None,
                   book_order: int = 0) -> List[Dict]:
        """
        Chunk all chapters in a book.

        Args:
            book_data: Book dictionary from parser
            series_name: Name of the series
            book_order: Order of this book in the series

        Returns:
            List of all chunks with metadata
        """
        all_chunks = []

        book_metadata = {
            "title": book_data["title"],
            "author": book_data["author"]
        }

        for chapter in book_data["chapters"]:
            chapter_chunks = self.chunk_chapter(
                chapter, book_metadata,
                series_name=series_name,
                book_order=book_order,
            )
            all_chunks.extend(chapter_chunks)

        logger.info(f"Created {len(all_chunks)} chunks from '{book_data['title']}'")
        return all_chunks

    def chunk_library(self, library_data: List[Dict],
                      series_manager=None) -> List[Dict]:
        """
        Chunk all books in a library.

        Args:
            library_data: List of book dictionaries
            series_manager: Optional SeriesManager for series/order metadata

        Returns:
            List of all chunks from all books
        """
        all_chunks = []

        for book_data in library_data:
            series_name = None
            book_order = 0
            if series_manager:
                series_name = series_manager.get_series_for_book(book_data["title"])
                book_order = series_manager.get_book_order(book_data["title"])

            book_chunks = self.chunk_book(
                book_data,
                series_name=series_name,
                book_order=book_order,
            )
            all_chunks.extend(book_chunks)

        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
