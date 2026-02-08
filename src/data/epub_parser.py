"""EPUB parsing utilities."""
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class EPUBParser:
    """Parse EPUB files to extract text and metadata."""

    def __init__(self):
        """Initialize parser."""
        pass

    def parse_epub(self, epub_path: Path) -> Dict:
        """
        Parse an EPUB file and extract text and metadata.

        Args:
            epub_path: Path to EPUB file

        Returns:
            Dictionary containing:
                - title: Book title
                - author: Book author
                - chapters: List of chapter dicts with text and metadata
        """
        try:
            book = epub.read_epub(str(epub_path))

            # Extract metadata
            metadata = self._extract_metadata(book)

            # Extract chapters
            chapters = self._extract_chapters(book)

            return {
                "title": metadata["title"],
                "author": metadata["author"],
                "chapters": chapters,
                "metadata": metadata,
                "source_file": str(epub_path)
            }
        except Exception as e:
            logger.error(f"Error parsing {epub_path}: {e}")
            raise

    def _extract_metadata(self, book: epub.EpubBook) -> Dict:
        """Extract metadata from EPUB."""
        metadata = {
            "title": "Unknown",
            "author": "Unknown",
            "language": "en",
            "identifier": ""
        }

        # Title
        title = book.get_metadata('DC', 'title')
        if title:
            metadata["title"] = title[0][0]

        # Author
        creator = book.get_metadata('DC', 'creator')
        if creator:
            metadata["author"] = creator[0][0]

        # Language
        language = book.get_metadata('DC', 'language')
        if language:
            metadata["language"] = language[0][0]

        # Identifier
        identifier = book.get_metadata('DC', 'identifier')
        if identifier:
            metadata["identifier"] = identifier[0][0]

        return metadata

    def _extract_chapters(self, book: epub.EpubBook) -> List[Dict]:
        """Extract chapter text from EPUB."""
        chapters = []

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                # Parse HTML content
                soup = BeautifulSoup(item.get_content(), 'html.parser')

                # Extract text
                text = soup.get_text(separator='\n', strip=True)

                # Skip empty chapters
                if not text or len(text.strip()) < 50:
                    continue

                # Try to extract chapter title
                chapter_title = self._extract_chapter_title(soup)

                chapters.append({
                    "title": chapter_title,
                    "text": text,
                    "item_id": item.get_id(),
                    "order": len(chapters)
                })

        return chapters

    def _extract_chapter_title(self, soup: BeautifulSoup) -> str:
        """Extract chapter title from HTML."""
        # Look for common title tags
        for tag in ['h1', 'h2', 'h3', 'title']:
            element = soup.find(tag)
            if element:
                title = element.get_text(strip=True)
                if title:
                    return title

        return f"Chapter {len([])}"

    def parse_library(self, library_dir: Path) -> List[Dict]:
        """
        Parse all EPUB files in a directory.

        Args:
            library_dir: Directory containing EPUB files

        Returns:
            List of parsed book dictionaries
        """
        library = []
        epub_files = list(library_dir.glob("**/*.epub"))

        logger.info(f"Found {len(epub_files)} EPUB files in {library_dir}")

        for epub_path in epub_files:
            try:
                logger.info(f"Parsing {epub_path.name}...")
                book_data = self.parse_epub(epub_path)
                library.append(book_data)
                logger.info(f"Successfully parsed {book_data['title']}")
            except Exception as e:
                logger.error(f"Failed to parse {epub_path.name}: {e}")
                continue

        return library
