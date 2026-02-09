"""Series detection and metadata management for book libraries."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SeriesManager:
    """Detect and manage book series relationships.

    Attempts auto-detection from EPUB metadata (Calibre series fields),
    falls back to manual configuration via series.json.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the series manager.

        Args:
            config_path: Path to series.json config file. If None, auto-detection only.
        """
        self.config_path = config_path
        self.series: Dict[str, dict] = {}  # series_name -> {author, books: [{title, order}]}
        self._book_to_series: Dict[str, str] = {}  # book_title -> series_name
        self._book_order: Dict[str, int] = {}  # book_title -> order_in_series

        if config_path and config_path.exists():
            self._load_config(config_path)

    def _load_config(self, config_path: Path):
        """Load series configuration from JSON file."""
        try:
            with open(config_path) as f:
                data = json.load(f)
            for series_entry in data.get("series", []):
                name = series_entry["name"]
                self.series[name] = {
                    "author": series_entry.get("author", "Unknown"),
                    "books": series_entry.get("books", []),
                }
                for book in series_entry.get("books", []):
                    self._book_to_series[book["title"]] = name
                    self._book_order[book["title"]] = book.get("order", 0)
            logger.info(f"Loaded {len(self.series)} series from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load series config: {e}")

    def save_config(self, config_path: Optional[Path] = None):
        """Save current series configuration to JSON."""
        path = config_path or self.config_path
        if path is None:
            logger.warning("No config path specified, cannot save")
            return

        data = {"series": []}
        for name, info in self.series.items():
            data["series"].append({
                "name": name,
                "author": info["author"],
                "books": info["books"],
            })

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved series config to {path}")

    def detect_series_from_books(self, library_data: List[Dict]):
        """Auto-detect series from parsed book metadata.

        Looks for Calibre-style series metadata fields that many EPUB
        management tools embed. Falls back to grouping by author when
        multiple books share the same author.

        Args:
            library_data: List of parsed book dictionaries from EPUBParser
        """
        calibre_detected: Dict[str, list] = {}

        for book in library_data:
            metadata = book.get("metadata", {})
            series_name = metadata.get("series")
            series_index = metadata.get("series_index")

            if series_name:
                if series_name not in calibre_detected:
                    calibre_detected[series_name] = []
                calibre_detected[series_name].append({
                    "title": book["title"],
                    "author": book["author"],
                    "order": int(float(series_index)) if series_index else 0,
                })

        # Register detected series (don't overwrite manually configured ones)
        for name, books in calibre_detected.items():
            if name not in self.series:
                books.sort(key=lambda b: b["order"])
                self.series[name] = {
                    "author": books[0]["author"] if books else "Unknown",
                    "books": [{"title": b["title"], "order": b["order"]} for b in books],
                }
                for b in books:
                    self._book_to_series[b["title"]] = name
                    self._book_order[b["title"]] = b["order"]
                logger.info(f"Auto-detected series: '{name}' ({len(books)} books)")

        # Group remaining books by author (2+ books from same author = possible series)
        author_groups: Dict[str, list] = {}
        for book in library_data:
            title = book["title"]
            if title not in self._book_to_series:
                author = book["author"]
                if author and author != "Unknown":
                    if author not in author_groups:
                        author_groups[author] = []
                    author_groups[author].append(title)

        for author, titles in author_groups.items():
            if len(titles) >= 2:
                group_name = f"{author} Collection"
                if group_name not in self.series:
                    books = [{"title": t, "order": i + 1} for i, t in enumerate(sorted(titles))]
                    self.series[group_name] = {
                        "author": author,
                        "books": books,
                    }
                    for b in books:
                        self._book_to_series[b["title"]] = group_name
                        self._book_order[b["title"]] = b["order"]
                    logger.info(f"Grouped by author: '{group_name}' ({len(titles)} books)")

        logger.info(
            f"Series detection complete: {len(self.series)} series, "
            f"{len(self._book_to_series)} books in series"
        )

    def get_series_for_book(self, book_title: str) -> Optional[str]:
        """Get the series name for a book title."""
        return self._book_to_series.get(book_title)

    def get_book_order(self, book_title: str) -> int:
        """Get the order of a book within its series (0 if unknown)."""
        return self._book_order.get(book_title, 0)

    def get_books_in_series(self, series_name: str) -> List[Dict]:
        """Get all books in a series, ordered."""
        series = self.series.get(series_name, {})
        books = series.get("books", [])
        return sorted(books, key=lambda b: b.get("order", 0))

    def get_all_series(self) -> Dict[str, dict]:
        """Get all series with their books."""
        return self.series

    def get_library_structure(self, library_data: List[Dict]) -> Dict:
        """Get the full library structure: series with books, plus ungrouped books.

        Args:
            library_data: List of parsed book dictionaries

        Returns:
            Dict with 'series' and 'ungrouped' keys
        """
        all_titles = {book["title"] for book in library_data}
        grouped_titles = set(self._book_to_series.keys())
        ungrouped_titles = all_titles - grouped_titles

        ungrouped = []
        for book in library_data:
            if book["title"] in ungrouped_titles:
                ungrouped.append({
                    "title": book["title"],
                    "author": book["author"],
                })

        return {
            "series": {
                name: {
                    "author": info["author"],
                    "books": info["books"],
                }
                for name, info in self.series.items()
            },
            "ungrouped": sorted(ungrouped, key=lambda b: b["title"]),
        }

    def compute_global_order(self, book_title: str, chapter_order: int,
                             chunk_index: int) -> int:
        """Compute a global ordering value for chronological sorting.

        Produces a single integer that orders chunks across an entire series:
        book_order * 100000 + chapter_order * 1000 + chunk_index

        Args:
            book_title: Title of the book
            chapter_order: Chapter number within the book
            chunk_index: Chunk index within the chapter

        Returns:
            Global ordering integer
        """
        book_order = self._book_order.get(book_title, 0)
        return book_order * 100000 + chapter_order * 1000 + chunk_index
