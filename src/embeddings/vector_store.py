"""Vector store using ChromaDB for metadata-aware retrieval."""
import numpy as np
import chromadb
from pathlib import Path
from typing import List, Dict, Optional
import logging
import uuid

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-based vector store for chunk embeddings.

    ChromaDB provides native metadata filtering, which enables scoped
    retrieval by book, series, chapter, or any other metadata field.
    """

    def __init__(self, embedding_dim: int, persist_dir: Optional[Path] = None):
        """
        Initialize vector store.

        Args:
            embedding_dim: Dimension of embeddings
            persist_dir: Directory for persistent storage. If None, uses in-memory.
        """
        self.embedding_dim = embedding_dim
        self._persist_dir = persist_dir

        if persist_dir:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_dir))
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name="library_chunks",
            metadata={"hnsw:space": "cosine"},
        )

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Dict]):
        """
        Add embeddings and metadata to the index.

        Args:
            embeddings: Array of embeddings (N, embedding_dim)
            chunks: List of chunk dictionaries with metadata
        """
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        embeddings_list = embeddings.astype("float32").tolist()

        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)

            # The child text is what we embed for retrieval
            documents.append(chunk.get("text", ""))

            # Build metadata dict (ChromaDB requires flat str/int/float/bool values)
            meta = {
                "book_title": chunk.get("book_title", ""),
                "book_author": chunk.get("book_author", ""),
                "chapter_title": chunk.get("chapter_title", ""),
                "chapter_order": int(chunk.get("chapter_order", 0)),
                "chunk_index": int(chunk.get("chunk_index", 0)),
                "total_chunks_in_chapter": int(chunk.get("total_chunks_in_chapter", 0)),
                "series_name": chunk.get("series_name", ""),
                "book_order_in_series": int(chunk.get("book_order_in_series", 0)),
                "global_order": int(chunk.get("global_order", 0)),
                "parent_id": chunk.get("parent_id", ""),
                "parent_text": chunk.get("parent_text", ""),
            }
            metadatas.append(meta)

        # ChromaDB has a batch limit; add in batches of 5000
        batch_size = 5000
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self._collection.add(
                ids=ids[start:end],
                embeddings=embeddings_list[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        logger.info(
            f"Added {len(embeddings)} embeddings. Total: {self._collection.count()}"
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               where: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar chunks, optionally filtered by metadata.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            where: Optional ChromaDB where filter for metadata,
                   e.g. {"book_title": "The Eye of the World"}
                   or {"series_name": "The Wheel of Time"}

        Returns:
            List of chunk dictionaries with similarity scores
        """
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_list = query_embedding.astype("float32").tolist()

        collection_count = self._collection.count()
        query_kwargs = {
            "query_embeddings": query_list,
            "n_results": min(top_k, collection_count) if collection_count > 0 else top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)

        # ChromaDB returns cosine *distance* (0 = identical, 2 = opposite).
        # Convert to cosine similarity for consistency with the rest of the pipeline.
        chunks = []
        if results["ids"] and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                distance = results["distances"][0][idx]
                similarity = 1.0 - distance  # cosine distance -> similarity

                meta = results["metadatas"][0][idx]
                chunk = {
                    "text": meta.get("parent_text") or results["documents"][0][idx],
                    "child_text": results["documents"][0][idx],
                    "book_title": meta.get("book_title", ""),
                    "book_author": meta.get("book_author", ""),
                    "chapter_title": meta.get("chapter_title", ""),
                    "chapter_order": meta.get("chapter_order", 0),
                    "chunk_index": meta.get("chunk_index", 0),
                    "total_chunks_in_chapter": meta.get("total_chunks_in_chapter", 0),
                    "series_name": meta.get("series_name", ""),
                    "book_order_in_series": meta.get("book_order_in_series", 0),
                    "global_order": meta.get("global_order", 0),
                    "parent_id": meta.get("parent_id", ""),
                    "similarity_score": float(similarity),
                }
                chunks.append(chunk)

        return chunks

    def save(self, save_dir: Path):
        """
        Persist the vector store to disk.

        If the store was created with a persist_dir, this is a no-op since
        ChromaDB PersistentClient auto-persists. Otherwise, re-creates
        as a PersistentClient at save_dir.

        Args:
            save_dir: Directory to save to
        """
        if self._persist_dir and str(self._persist_dir) == str(save_dir):
            logger.info(f"Vector store already persisted at {save_dir}")
            return

        if self._persist_dir:
            logger.info(f"Vector store already persisted at {self._persist_dir}")
            return

        # In-memory store: need to re-create as persistent
        save_dir.mkdir(parents=True, exist_ok=True)
        new_client = chromadb.PersistentClient(path=str(save_dir))
        new_collection = new_client.get_or_create_collection(
            name="library_chunks",
            metadata={"hnsw:space": "cosine"},
        )

        # Copy data from in-memory to persistent
        count = self._collection.count()
        if count > 0:
            all_data = self._collection.get(
                include=["embeddings", "documents", "metadatas"]
            )
            batch_size = 5000
            for start in range(0, len(all_data["ids"]), batch_size):
                end = start + batch_size
                new_collection.add(
                    ids=all_data["ids"][start:end],
                    embeddings=all_data["embeddings"][start:end],
                    documents=all_data["documents"][start:end],
                    metadatas=all_data["metadatas"][start:end],
                )

        self._client = new_client
        self._collection = new_collection
        self._persist_dir = save_dir
        logger.info(f"Saved vector store to {save_dir}")

    @classmethod
    def load(cls, load_dir: Path) -> "VectorStore":
        """
        Load a persisted vector store from disk.

        Args:
            load_dir: Directory to load from

        Returns:
            VectorStore instance
        """
        client = chromadb.PersistentClient(path=str(load_dir))
        collection = client.get_or_create_collection(
            name="library_chunks",
            metadata={"hnsw:space": "cosine"},
        )

        # Infer embedding dimension from a sample
        embedding_dim = 768  # default
        if collection.count() > 0:
            sample = collection.get(limit=1, include=["embeddings"])
            if sample["embeddings"]:
                embedding_dim = len(sample["embeddings"][0])

        store = cls.__new__(cls)
        store.embedding_dim = embedding_dim
        store._persist_dir = load_dir
        store._client = client
        store._collection = collection

        logger.info(
            f"Loaded vector store from {load_dir} with {collection.count()} vectors"
        )
        return store

    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        count = self._collection.count()
        return {
            "total_vectors": count,
            "embedding_dim": self.embedding_dim,
            "total_chunks": count,
        }

    def get_chunks_by_book(self, book_title: str) -> List[Dict]:
        """Get all chunks for a specific book, sorted by chapter and chunk order.

        Args:
            book_title: The book title to retrieve chunks for

        Returns:
            List of chunk dicts sorted chronologically
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.get(
            where={"book_title": book_title},
            include=["documents", "metadatas"],
        )

        chunks = []
        for i, meta in enumerate(results["metadatas"]):
            chunk = {
                "text": meta.get("parent_text") or results["documents"][i],
                "book_title": meta.get("book_title", ""),
                "book_author": meta.get("book_author", ""),
                "chapter_title": meta.get("chapter_title", ""),
                "chapter_order": meta.get("chapter_order", 0),
                "chunk_index": meta.get("chunk_index", 0),
                "series_name": meta.get("series_name", ""),
                "book_order_in_series": meta.get("book_order_in_series", 0),
                "global_order": meta.get("global_order", 0),
                "parent_id": meta.get("parent_id", ""),
            }
            chunks.append(chunk)

        # Deduplicate by parent_id (keep one child per parent)
        seen_parents = set()
        unique = []
        for c in chunks:
            pid = c.get("parent_id", "")
            if pid and pid in seen_parents:
                continue
            if pid:
                seen_parents.add(pid)
            unique.append(c)

        # Sort by chapter_order then chunk_index
        unique.sort(key=lambda c: (c["chapter_order"], c["chunk_index"]))
        return unique

    def get_unique_books(self) -> List[Dict]:
        """Get a list of unique books in the index with their metadata.

        Returns:
            List of dicts with book_title, book_author, series_name, book_order
        """
        if self._collection.count() == 0:
            return []

        # Retrieve all metadata to extract unique books
        all_meta = self._collection.get(include=["metadatas"])
        seen = {}
        for meta in all_meta["metadatas"]:
            title = meta.get("book_title", "")
            if title and title not in seen:
                seen[title] = {
                    "book_title": title,
                    "book_author": meta.get("book_author", ""),
                    "series_name": meta.get("series_name", ""),
                    "book_order_in_series": meta.get("book_order_in_series", 0),
                }
        return sorted(seen.values(), key=lambda b: b["book_title"])

    def get_unique_series(self) -> Dict:
        """Get a list of unique series with their books.

        Returns:
            Dict with 'series' list and 'ungrouped' list
        """
        books = self.get_unique_books()
        series_map: Dict[str, dict] = {}
        ungrouped = []

        for book in books:
            sname = book.get("series_name", "")
            if sname:
                if sname not in series_map:
                    series_map[sname] = {
                        "series_name": sname,
                        "author": book["book_author"],
                        "books": [],
                    }
                series_map[sname]["books"].append({
                    "title": book["book_title"],
                    "order": book["book_order_in_series"],
                })
            else:
                ungrouped.append(book)

        # Sort books within each series
        for s in series_map.values():
            s["books"].sort(key=lambda b: b["order"])

        # Return series as a dict mapping name -> book list for easy UI rendering
        series_dict = {}
        for s in series_map.values():
            series_dict[s["series_name"]] = s["books"]

        return {
            "series": series_dict,
            "ungrouped": ungrouped,
        }
