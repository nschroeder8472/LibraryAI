"""Summary generation pipeline using map-reduce over book chunks."""
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from ..embeddings.vector_store import VectorStore
from ..generation.model import create_language_model, BaseLanguageModel
from ..generation.prompt_templates import PromptTemplates
from ..config import config

logger = logging.getLogger(__name__)


class SummaryPipeline:
    """Generate book and series summaries via map-reduce.

    The pipeline works in two stages:
    1. **Map**: Summarize each chapter independently.
    2. **Reduce**: Combine chapter summaries into a book summary.

    For series summaries, book summaries are combined into an overview.
    Generated summaries are cached to disk so they only need to be
    generated once.
    """

    def __init__(self, vector_store: VectorStore,
                 llm: Optional[BaseLanguageModel] = None):
        self.vector_store = vector_store
        self.llm = llm or create_language_model(
            backend=config.generation.backend,
            model_name=config.generation.model_name,
            device=config.generation.device,
            max_new_tokens=config.generation.max_new_tokens,
            temperature=config.generation.temperature,
            top_p=config.generation.top_p,
            do_sample=config.generation.do_sample,
            use_8bit=config.generation.use_8bit,
            use_4bit=config.generation.use_4bit,
        )
        self.templates = PromptTemplates()

        # Cache directory
        self._cache_dir = config.data.data_dir / "summaries"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def summarize_book(self, book_title: str,
                       force: bool = False) -> Dict:
        """Generate a summary for a single book.

        Args:
            book_title: Title of the book to summarize
            force: If True, regenerate even if cached

        Returns:
            Dict with 'summary', 'book_title', 'book_author',
            'chapter_summaries'
        """
        # Check cache
        cache_key = self._cache_key("book", book_title)
        if not force:
            cached = self._load_cache(cache_key)
            if cached:
                logger.info(f"Using cached summary for '{book_title}'")
                return cached

        logger.info(f"Generating summary for '{book_title}'")

        # Get all chunks for this book, ordered by chapter
        chunks = self.vector_store.get_chunks_by_book(book_title)
        if not chunks:
            return {
                "summary": f"No content found for '{book_title}' in the index.",
                "book_title": book_title,
                "book_author": "",
                "chapter_summaries": [],
            }

        book_author = chunks[0].get("book_author", "Unknown")

        # Group chunks by chapter
        chapters: Dict[str, list] = {}
        chapter_order_map: Dict[str, int] = {}
        for chunk in chunks:
            ch_title = chunk.get("chapter_title", "Unknown Chapter")
            if ch_title not in chapters:
                chapters[ch_title] = []
                chapter_order_map[ch_title] = chunk.get("chapter_order", 0)
            chapters[ch_title].append(chunk["text"])

        # Sort chapters by order
        sorted_chapters = sorted(
            chapters.keys(),
            key=lambda c: chapter_order_map.get(c, 0),
        )

        # Map: summarize each chapter
        chapter_summaries = []
        for i, ch_title in enumerate(sorted_chapters):
            logger.info(
                f"Summarizing chapter {i+1}/{len(sorted_chapters)}: {ch_title}"
            )
            chapter_text = "\n\n".join(chapters[ch_title])

            # Truncate very long chapters to fit model context
            if len(chapter_text) > 6000:
                chapter_text = chapter_text[:6000] + "\n[...truncated]"

            prompt = self.templates.chapter_summary_prompt(
                book_title, ch_title, chapter_text
            )
            summary = self.llm.generate(prompt)
            chapter_summaries.append({
                "chapter_title": ch_title,
                "summary": summary.strip(),
            })

        # Reduce: combine chapter summaries into book summary
        logger.info(f"Combining {len(chapter_summaries)} chapter summaries")
        combined = "\n\n".join(
            f"Chapter: {cs['chapter_title']}\n{cs['summary']}"
            for cs in chapter_summaries
        )

        # Truncate if combined summaries are too long
        if len(combined) > 6000:
            combined = combined[:6000] + "\n[...truncated]"

        prompt = self.templates.book_summary_prompt(
            book_title, book_author, combined
        )
        book_summary = self.llm.generate(prompt)

        result = {
            "summary": book_summary.strip(),
            "book_title": book_title,
            "book_author": book_author,
            "chapter_summaries": chapter_summaries,
        }

        # Cache the result
        self._save_cache(cache_key, result)

        logger.info(f"Summary complete for '{book_title}'")
        return result

    def summarize_series(self, series_name: str,
                         force: bool = False) -> Dict:
        """Generate a summary for a book series.

        Args:
            series_name: Name of the series
            force: If True, regenerate even if cached

        Returns:
            Dict with 'summary', 'series_name', 'book_summaries'
        """
        # Check cache
        cache_key = self._cache_key("series", series_name)
        if not force:
            cached = self._load_cache(cache_key)
            if cached:
                logger.info(f"Using cached summary for series '{series_name}'")
                return cached

        logger.info(f"Generating summary for series '{series_name}'")

        # Get all books in this series
        library = self.vector_store.get_unique_series()
        series_data = library.get("series", {})
        books = series_data.get(series_name, [])

        if not books:
            return {
                "summary": f"No books found for series '{series_name}'.",
                "series_name": series_name,
                "book_summaries": [],
            }

        # Summarize each book
        book_summaries = []
        for book in books:
            title = book.get("title", book) if isinstance(book, dict) else book
            book_result = self.summarize_book(title, force=force)
            book_summaries.append({
                "book_title": title,
                "summary": book_result["summary"],
            })

        # Combine book summaries into series overview
        combined = "\n\n".join(
            f"Book: {bs['book_title']}\n{bs['summary']}"
            for bs in book_summaries
        )

        if len(combined) > 6000:
            combined = combined[:6000] + "\n[...truncated]"

        prompt = self.templates.series_summary_prompt(series_name, combined)
        series_summary = self.llm.generate(prompt)

        result = {
            "summary": series_summary.strip(),
            "series_name": series_name,
            "book_summaries": book_summaries,
        }

        self._save_cache(cache_key, result)

        logger.info(f"Summary complete for series '{series_name}'")
        return result

    def _cache_key(self, kind: str, name: str) -> str:
        """Generate a cache filename."""
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in name)
        return f"{kind}_{safe.strip()}"

    def _load_cache(self, key: str) -> Optional[Dict]:
        """Load a cached summary."""
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                return None
        return None

    def _save_cache(self, key: str, data: Dict):
        """Save a summary to cache."""
        path = self._cache_dir / f"{key}.json"
        try:
            path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Could not cache summary: {e}")
