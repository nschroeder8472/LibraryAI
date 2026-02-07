# LibraryAI Implementation Plan

## Project Overview

LibraryAI is a Retrieval-Augmented Generation (RAG) system designed to answer questions about a personal ebook library. The MVP focuses on EPUB files, with future expansion to audiobooks, PDFs, and other formats.

### Core Capabilities
- Parse and index EPUB files from a personal library
- Answer questions like "What chapter contains passage X?"
- Provide reading recommendations based on listening/reading history
- Retrieve relevant context from books to answer queries

### Technology Stack
- **Base Model**: Meta's Llama 3.2-1B (lightweight, fine-tunable)
- **Framework**: Hugging Face transformers
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **EPUB Parsing**: ebooklib
- **Text Processing**: Beautiful Soup 4, langchain text splitters

---

## Architecture Overview

### RAG Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     USER QUERY                              │
│              "What chapter discusses X?"                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  QUERY PROCESSOR                            │
│  - Validates input                                          │
│  - Extracts intent                                          │
│  - Adds context (reading history if needed)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               RETRIEVAL COMPONENT                           │
│                                                             │
│  ┌────────────────┐         ┌─────────────────┐             │
│  │ Query Embedding│────────▶│ Vector Search   │             │
│  │ (Sentence-BERT)│         │ (FAISS Index)   │             │
│  └────────────────┘         └────────┬────────┘             │
│                                      │                      │
│                                      ▼                      │
│                            ┌─────────────────┐              │
│                            │  Top-K Chunks   │              │
│                            │  (with metadata)│              │
│                            └─────────────────┘              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              GENERATION COMPONENT                           │
│                                                             │
│  ┌──────────────────────────────────────────────┐           │
│  │  Prompt Template:                            │           │
│  │  - System instructions                       │           │
│  │  - Retrieved context chunks                  │           │
│  │  - User query                                │           │
│  │  - (Optional) Reading history                │           │
│  └────────────────────┬─────────────────────────┘           │
│                       │                                     │
│                       ▼                                     │
│            ┌──────────────────────┐                         │
│            │  Llama 3.2-1B Model  │                         │
│            │  (Fine-tuned)        │                         │
│            └──────────┬───────────┘                         │
│                       │                                     │
│                       ▼                                     │
│            ┌──────────────────────┐                         │
│            │  Generated Answer    │                         │
│            │  with Citations      │                         │
│            └──────────────────────┘                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    RESPONSE TO USER                         │
│  "The concept appears in Chapter 5 of 'Book Title',         │
│   specifically in the section about..."                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  OFFLINE INDEXING PIPELINE                  │
│  (Runs once per book or when library updates)               │
│                                                             │
│  ┌────────────┐    ┌──────────┐    ┌──────────────┐         │
│  │ EPUB Files │───▶│  Parser  │───▶│   Chunker    │         │
│  │ (data/raw/)│    │(ebooklib)│    │ (Recursive)  │         │
│  └────────────┘    └──────────┘    └──────┬───────┘         │
│                                           │                 │
│                                           ▼                 │
│                                    ┌────────────────┐       │
│                                    │  Text Chunks   │       │
│                                    │  + Metadata    │       │
│                                    └────────┬───────┘       │
│                                             │               │
│                                             ▼               │
│                                    ┌────────────────┐       │
│                                    │   Embedder     │       │
│                                    │(Sentence-BERT) │       │
│                                    └────────┬───────┘       │
│                                             │               │
│                                             ▼               │
│                                    ┌────────────────┐       │
│                                    │  Vector Store  │       │
│                                    │  (FAISS Index) │       │
│                                    └────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Indexing Phase** (Offline):
   - EPUB files → Parse to extract text → Chunk into passages → Generate embeddings → Store in FAISS

2. **Query Phase** (Runtime):
   - User query → Embed query → Retrieve top-K similar chunks → Construct prompt → Generate answer with Llama

---

## Project Structure

```
LibraryAI/
├── data/
│   ├── raw/                    # Original EPUB files (gitignored)
│   ├── processed/              # Extracted and chunked text (gitignored)
│   └── vector_store/           # FAISS indices (gitignored)
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── data/
│   │   ├── __init__.py
│   │   ├── epub_parser.py      # EPUB parsing logic
│   │   ├── chunker.py          # Text chunking strategies
│   │   └── metadata.py         # Metadata extraction
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── embedder.py         # Embedding generation
│   │   └── vector_store.py     # FAISS operations
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py        # Search and retrieval logic
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── model.py            # Llama model loading
│   │   └── prompt_templates.py # Prompt engineering
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── indexing.py         # Indexing orchestration
│   │   └── query.py            # Query pipeline
│   └── utils/
│       ├── __init__.py
│       └── logging_config.py   # Logging setup
├── scripts/
│   ├── build_index.py          # Index library books
│   └── query_cli.py            # Interactive CLI for queries
├── tests/
│   ├── test_parser.py
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_generation.py
├── plan/
│   └── project_plan.md         # This file
├── main.py                     # Entry point
├── requirements.txt            # Dependencies
├── .gitignore
├── CLAUDE.md                   # Project context for Claude
└── README.md                   # User documentation
```

---

## Detailed Implementation Guide

### 1. Configuration Management (`src/config.py`)

Central configuration for all components.

```python
"""Configuration management for LibraryAI."""
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataConfig:
    """Data paths configuration."""
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    vector_store_dir: Path = Path("data/vector_store")

    def __post_init__(self):
        """Create directories if they don't exist."""
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class ChunkingConfig:
    """Text chunking configuration."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: list = None

    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", ". ", " ", ""]

@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"  # or "cuda" if available
    batch_size: int = 32

@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    top_k: int = 5
    similarity_threshold: float = 0.7

@dataclass
class GenerationConfig:
    """Language model configuration."""
    model_name: str = "meta-llama/Llama-3.2-1B"
    device: str = "cpu"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = DataConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()

# Global config instance
config = Config()
```

---

### 2. EPUB Parsing (`src/data/epub_parser.py`)

Extract text and metadata from EPUB files.

```python
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
        epub_files = list(library_dir.glob("*.epub"))

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
```

---

### 3. Text Chunking (`src/data/chunker.py`)

Split extracted text into manageable chunks for embedding.

```python
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
```

---

### 4. Embedding Generation (`src/embeddings/embedder.py`)

Generate vector embeddings for text chunks.

```python
"""Embedding generation utilities."""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import torch
import logging

logger = logging.getLogger(__name__)

class Embedder:
    """Generate embeddings for text using sentence transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu", batch_size: int = 32):
        """
        Initialize embedder.

        Args:
            model_name: Name of sentence-transformer model
            device: Device to use (cpu or cuda)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embeddings (num_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    def embed_chunks(self, chunks: List[Dict]) -> tuple[np.ndarray, List[Dict]]:
        """
        Generate embeddings for chunks with metadata.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Tuple of (embeddings array, chunks with metadata)
        """
        texts = [chunk["text"] for chunk in chunks]

        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embed_batch(texts)

        # Attach embedding to each chunk
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        return embeddings, chunks
```

---

### 5. Vector Store (`src/embeddings/vector_store.py`)

Manage FAISS index for similarity search.

```python
"""Vector store using FAISS."""
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """FAISS-based vector store for chunk embeddings."""

    def __init__(self, embedding_dim: int):
        """
        Initialize vector store.

        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
        self.chunks_metadata = []  # Store chunk metadata
        self.is_trained = False

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Dict]):
        """
        Add embeddings and metadata to the index.

        Args:
            embeddings: Array of embeddings (N, embedding_dim)
            chunks: List of chunk dictionaries with metadata
        """
        # Ensure correct shape
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))

        # Store metadata
        self.chunks_metadata.extend(chunks)

        logger.info(f"Added {len(embeddings)} embeddings. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of chunk dictionaries with similarity scores
        """
        # Ensure correct shape
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search FAISS index
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            top_k
        )

        # Retrieve metadata and add scores
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks_metadata):
                chunk = self.chunks_metadata[idx].copy()
                # Convert L2 distance to similarity score (lower is better)
                chunk["similarity_score"] = float(dist)
                results.append(chunk)

        return results

    def save(self, save_dir: Path):
        """
        Save index and metadata to disk.

        Args:
            save_dir: Directory to save to
        """
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = save_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))

        # Save metadata
        metadata_path = save_dir / "chunks_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.chunks_metadata, f)

        logger.info(f"Saved vector store to {save_dir}")

    @classmethod
    def load(cls, load_dir: Path) -> 'VectorStore':
        """
        Load index and metadata from disk.

        Args:
            load_dir: Directory to load from

        Returns:
            VectorStore instance
        """
        # Load FAISS index
        index_path = load_dir / "faiss_index.bin"
        index = faiss.read_index(str(index_path))

        # Load metadata
        metadata_path = load_dir / "chunks_metadata.pkl"
        with open(metadata_path, 'rb') as f:
            chunks_metadata = pickle.load(f)

        # Create instance
        embedding_dim = index.d
        vector_store = cls(embedding_dim)
        vector_store.index = index
        vector_store.chunks_metadata = chunks_metadata
        vector_store.is_trained = True

        logger.info(f"Loaded vector store from {load_dir} with {index.ntotal} vectors")

        return vector_store

    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "total_chunks": len(self.chunks_metadata)
        }
```

---

### 6. Retriever (`src/retrieval/retriever.py`)

Orchestrate retrieval pipeline.

```python
"""Retrieval component."""
from typing import List, Dict
from ..embeddings.embedder import Embedder
from ..embeddings.vector_store import VectorStore
import logging

logger = logging.getLogger(__name__)

class Retriever:
    """Retrieve relevant chunks for a query."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore,
                 top_k: int = 5, similarity_threshold: float = 0.7):
        """
        Initialize retriever.

        Args:
            embedder: Embedder instance
            vector_store: VectorStore instance
            top_k: Number of results to retrieve
            similarity_threshold: Minimum similarity score
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User query text

        Returns:
            List of relevant chunks with metadata and scores
        """
        # Embed query
        logger.info(f"Embedding query: {query[:50]}...")
        query_embedding = self.embedder.embed_text(query)

        # Search vector store
        results = self.vector_store.search(query_embedding, self.top_k)

        # Filter by threshold (lower L2 distance = more similar)
        # Note: FAISS L2 returns squared distances
        filtered_results = [
            r for r in results
            if r["similarity_score"] < self.similarity_threshold
        ]

        logger.info(f"Retrieved {len(filtered_results)} chunks above threshold")

        return filtered_results

    def format_context(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into context string for prompt.

        Args:
            chunks: Retrieved chunks

        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant information found in the library."

        context_parts = []
        for idx, chunk in enumerate(chunks, 1):
            context_part = f"""
Context {idx}:
Book: "{chunk['book_title']}" by {chunk['book_author']}
Chapter: {chunk['chapter_title']}
Text: {chunk['text']}
---"""
            context_parts.append(context_part)

        return "\n".join(context_parts)
```

---

### 7. Language Model (`src/generation/model.py`)

Load and manage Llama model.

```python
"""Language model utilities."""
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import logging

logger = logging.getLogger(__name__)

class LanguageModel:
    """Llama language model for generation."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B",
                 device: str = "cpu", max_new_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 0.9):
        """
        Initialize language model.

        Args:
            model_name: Hugging Face model name
            device: Device to use
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        logger.info(f"Loading model: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        self.model.to(device)
        self.model.eval()

        logger.info(f"Model loaded on {device}")

    def generate(self, prompt: str) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text.strip()
```

---

### 8. Prompt Templates (`src/generation/prompt_templates.py`)

Engineered prompts for different query types.

```python
"""Prompt templates for RAG system."""
from typing import List, Dict, Optional

class PromptTemplates:
    """Collection of prompt templates."""

    @staticmethod
    def qa_prompt(query: str, context: str) -> str:
        """
        Question answering prompt.

        Args:
            query: User question
            context: Retrieved context

        Returns:
            Formatted prompt
        """
        prompt = f"""You are a helpful assistant answering questions about a personal book library.

Context from the library:
{context}

Question: {query}

Instructions:
- Answer the question based ONLY on the provided context
- If the context contains the answer, provide it clearly and cite the book and chapter
- If the context doesn't contain enough information, say so honestly
- Be concise but complete

Answer:"""
        return prompt

    @staticmethod
    def recommendation_prompt(query: str, context: str,
                            reading_history: Optional[str] = None) -> str:
        """
        Book recommendation prompt.

        Args:
            query: User request for recommendations
            context: Retrieved context from similar books
            reading_history: Recent reading history

        Returns:
            Formatted prompt
        """
        history_section = ""
        if reading_history:
            history_section = f"\nRecent reading history:\n{reading_history}\n"

        prompt = f"""You are a helpful assistant recommending books from a personal library.

{history_section}
Relevant books from the library:
{context}

Request: {query}

Instructions:
- Recommend books from the library based on the request
- Consider the reading history if provided
- Explain why each recommendation fits the request
- Limit to 3-5 recommendations

Recommendations:"""
        return prompt

    @staticmethod
    def passage_location_prompt(query: str, context: str) -> str:
        """
        Find passage location prompt.

        Args:
            query: Query with passage to locate
            context: Retrieved context chunks

        Returns:
            Formatted prompt
        """
        prompt = f"""You are a helpful assistant locating passages in a book library.

Context from the library:
{context}

Query: {query}

Instructions:
- Identify which book and chapter contains the passage or concept
- Provide specific location information (book title, author, chapter)
- If found in multiple places, list all occurrences
- If not found, say so clearly

Location:"""
        return prompt
```

---

### 9. Indexing Pipeline (`src/pipeline/indexing.py`)

Orchestrate the indexing process.

```python
"""Indexing pipeline orchestration."""
from pathlib import Path
from typing import List, Dict
import json
import logging

from ..data.epub_parser import EPUBParser
from ..data.chunker import TextChunker
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
            separators=config.chunking.separators
        )
        self.embedder = Embedder(
            model_name=config.embedding.model_name,
            device=config.embedding.device,
            batch_size=config.embedding.batch_size
        )
        self.vector_store = VectorStore(self.embedder.embedding_dim)

    def index_library(self, library_dir: Path = None) -> VectorStore:
        """
        Index all books in library directory.

        Args:
            library_dir: Directory containing EPUB files

        Returns:
            Populated vector store
        """
        if library_dir is None:
            library_dir = config.data.raw_data_dir

        logger.info(f"Starting indexing pipeline for {library_dir}")

        # Step 1: Parse EPUBs
        logger.info("Step 1: Parsing EPUB files...")
        library_data = self.parser.parse_library(library_dir)
        logger.info(f"Parsed {len(library_data)} books")

        # Save parsed data
        parsed_path = config.data.processed_data_dir / "parsed_books.json"
        with open(parsed_path, 'w') as f:
            # Remove embedding data for JSON serialization
            json_safe_data = [{k: v for k, v in book.items() if k != 'embedding'}
                             for book in library_data]
            json.dump(json_safe_data, f, indent=2)
        logger.info(f"Saved parsed data to {parsed_path}")

        # Step 2: Chunk text
        logger.info("Step 2: Chunking text...")
        all_chunks = self.chunker.chunk_library(library_data)
        logger.info(f"Created {len(all_chunks)} chunks")

        # Step 3: Generate embeddings
        logger.info("Step 3: Generating embeddings...")
        embeddings, chunks_with_embeddings = self.embedder.embed_chunks(all_chunks)

        # Step 4: Build vector store
        logger.info("Step 4: Building vector store...")
        self.vector_store.add_embeddings(embeddings, chunks_with_embeddings)

        # Step 5: Save vector store
        logger.info("Step 5: Saving vector store...")
        self.vector_store.save(config.data.vector_store_dir)

        logger.info("Indexing pipeline complete!")
        logger.info(f"Stats: {self.vector_store.get_stats()}")

        return self.vector_store
```

---

### 10. Query Pipeline (`src/pipeline/query.py`)

Orchestrate the query answering process.

```python
"""Query pipeline orchestration."""
from typing import Dict, Optional
import logging

from ..retrieval.retriever import Retriever
from ..generation.model import LanguageModel
from ..generation.prompt_templates import PromptTemplates
from ..embeddings.embedder import Embedder
from ..embeddings.vector_store import VectorStore
from ..config import config

logger = logging.getLogger(__name__)

class QueryPipeline:
    """Orchestrate the RAG query pipeline."""

    def __init__(self, vector_store: VectorStore):
        """
        Initialize query pipeline.

        Args:
            vector_store: Loaded vector store
        """
        self.embedder = Embedder(
            model_name=config.embedding.model_name,
            device=config.embedding.device
        )

        self.retriever = Retriever(
            embedder=self.embedder,
            vector_store=vector_store,
            top_k=config.retrieval.top_k,
            similarity_threshold=config.retrieval.similarity_threshold
        )

        self.llm = LanguageModel(
            model_name=config.generation.model_name,
            device=config.generation.device,
            max_new_tokens=config.generation.max_new_tokens,
            temperature=config.generation.temperature,
            top_p=config.generation.top_p
        )

        self.templates = PromptTemplates()

    def query(self, query: str, query_type: str = "qa",
             reading_history: Optional[str] = None) -> Dict:
        """
        Process a query through the RAG pipeline.

        Args:
            query: User query
            query_type: Type of query (qa, recommendation, passage_location)
            reading_history: Optional reading history for recommendations

        Returns:
            Dictionary with answer and retrieved contexts
        """
        logger.info(f"Processing query: {query[:50]}...")

        # Step 1: Retrieve relevant chunks
        logger.info("Retrieving relevant context...")
        retrieved_chunks = self.retriever.retrieve(query)

        if not retrieved_chunks:
            return {
                "answer": "I couldn't find relevant information in your library to answer this question.",
                "contexts": [],
                "query": query
            }

        # Step 2: Format context
        context = self.retriever.format_context(retrieved_chunks)

        # Step 3: Select prompt template
        if query_type == "recommendation":
            prompt = self.templates.recommendation_prompt(
                query, context, reading_history
            )
        elif query_type == "passage_location":
            prompt = self.templates.passage_location_prompt(query, context)
        else:  # default to qa
            prompt = self.templates.qa_prompt(query, context)

        # Step 4: Generate answer
        logger.info("Generating answer...")
        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "contexts": retrieved_chunks,
            "query": query,
            "query_type": query_type
        }
```

---

## Dependencies (`requirements.txt`)

```
# Core ML frameworks
torch>=2.0.0
transformers>=4.35.0
sentence-transformers>=2.2.0

# Vector store
faiss-cpu>=1.7.4  # Use faiss-gpu if CUDA available

# Text processing
langchain>=0.1.0
beautifulsoup4>=4.12.0
lxml>=4.9.0

# EPUB parsing
ebooklib>=0.18

# Utilities
numpy>=1.24.0
```

---

## Implementation Order & Milestones

### Phase 1: Foundation (Days 1-2)
**Goal**: Set up project structure and configuration

- [ ] Create project directory structure
- [ ] Implement `src/config.py` with all configuration classes
- [ ] Set up `src/utils/logging_config.py`
- [ ] Install dependencies
- [ ] Add sample EPUB file to `data/raw/`

**Verification**: Run `python -c "from src.config import config; print(config)"` successfully

---

### Phase 2: Data Pipeline (Days 3-5)
**Goal**: Parse and chunk EPUB files

- [ ] Implement `src/data/epub_parser.py`
- [ ] Write unit tests for parser (`tests/test_parser.py`)
- [ ] Implement `src/data/chunker.py`
- [ ] Write unit tests for chunker (`tests/test_chunker.py`)

**Verification**:
```python
from src.data.epub_parser import EPUBParser
from src.data.chunker import TextChunker

parser = EPUBParser()
book = parser.parse_epub("data/raw/sample.epub")
print(f"Parsed: {book['title']} with {len(book['chapters'])} chapters")

chunker = TextChunker()
chunks = chunker.chunk_book(book)
print(f"Created {len(chunks)} chunks")
```

---

### Phase 3: Embeddings & Vector Store (Days 6-8)
**Goal**: Generate embeddings and build searchable index

- [ ] Implement `src/embeddings/embedder.py`
- [ ] Implement `src/embeddings/vector_store.py`
- [ ] Write unit tests (`tests/test_embeddings.py`)

**Verification**:
```python
from src.embeddings.embedder import Embedder
from src.embeddings.vector_store import VectorStore

embedder = Embedder()
embeddings, chunks = embedder.embed_chunks(chunks)
print(f"Generated embeddings: {embeddings.shape}")

vs = VectorStore(embedder.embedding_dim)
vs.add_embeddings(embeddings, chunks)
vs.save("data/vector_store")

# Test search
query_emb = embedder.embed_text("test query")
results = vs.search(query_emb, top_k=3)
print(f"Found {len(results)} results")
```

---

### Phase 4: Retrieval (Days 9-10)
**Goal**: Implement retrieval pipeline

- [ ] Implement `src/retrieval/retriever.py`
- [ ] Write unit tests (`tests/test_retriever.py`)

**Verification**:
```python
from src.retrieval.retriever import Retriever

vs = VectorStore.load("data/vector_store")
retriever = Retriever(embedder, vs, top_k=5)

results = retriever.retrieve("What is the main theme?")
print(f"Retrieved {len(results)} chunks")
print(retriever.format_context(results))
```

---

### Phase 5: Generation (Days 11-14)
**Goal**: Implement LLM generation

- [ ] Implement `src/generation/model.py`
- [ ] Implement `src/generation/prompt_templates.py`
- [ ] Write unit tests (`tests/test_generation.py`)

**Verification**:
```python
from src.generation.model import LanguageModel
from src.generation.prompt_templates import PromptTemplates

llm = LanguageModel()
templates = PromptTemplates()

prompt = templates.qa_prompt("Test question?", "Test context")
answer = llm.generate(prompt)
print(f"Generated: {answer}")
```

---

### Phase 6: Pipeline Integration (Days 15-17)
**Goal**: Orchestrate full pipelines

- [ ] Implement `src/pipeline/indexing.py`
- [ ] Implement `src/pipeline/query.py`
- [ ] Create `scripts/build_index.py`
- [ ] Create `scripts/query_cli.py`

**Verification**:
```bash
# Build index
python scripts/build_index.py

# Query
python scripts/query_cli.py "What books do I have about AI?"
```

---

### Phase 7: CLI & Integration (Days 18-20)
**Goal**: Create user-facing interface

- [ ] Implement main.py with CLI interface
- [ ] Add interactive query mode
- [ ] Add batch query mode
- [ ] Write integration tests

**Verification**:
```bash
python main.py index --library-dir data/raw
python main.py query "What is the main theme of book X?"
python main.py interactive
```

---

### Phase 8: Testing & Polish (Days 21-23)
**Goal**: Comprehensive testing and documentation

- [ ] Write end-to-end tests
- [ ] Test with multiple books
- [ ] Optimize performance (batching, caching)
- [ ] Write README.md
- [ ] Document API

**Verification**: All tests pass, system works end-to-end

---

## Testing Strategy

### Unit Tests
Test each module in isolation:
- Parser: Test with valid/invalid EPUBs
- Chunker: Test chunk sizes, overlaps
- Embedder: Test embedding generation
- Vector Store: Test add/search/save/load
- Retriever: Test retrieval quality
- LLM: Test generation

### Integration Tests
Test component interactions:
- Parser → Chunker → Embedder → Vector Store
- Retriever → LLM
- Full query pipeline

### End-to-End Tests
Test complete workflows:
- Index library → Query → Get answer
- Multiple books, different query types
- Edge cases (empty library, no results)

---

## Future Enhancements

### Phase 9: Fine-tuning (Optional)
**Goal**: Fine-tune Llama for better library-specific answers

1. **Collect Training Data**:
   - Generate question-answer pairs from books
   - Use synthetic data generation
   - Format as instruction-tuning dataset

2. **Fine-tuning Setup**:
   - Use PEFT (LoRA) for efficient fine-tuning
   - Create training script with Hugging Face Trainer
   - Define evaluation metrics

3. **Training**:
   ```python
   from peft import LoraConfig, get_peft_model

   lora_config = LoraConfig(
       r=8,
       lora_alpha=16,
       target_modules=["q_proj", "v_proj"],
       lora_dropout=0.05,
       bias="none",
       task_type="CAUSAL_LM"
   )
   ```

### Additional Features
- **Multi-format Support**: Add PDF, audiobook transcripts
- **Advanced Chunking**: Semantic chunking, chapter-aware splitting
- **Query Understanding**: Intent classification, query expansion
- **Answer Quality**: Add re-ranking, answer verification
- **Web Interface**: Flask/Streamlit frontend
- **Multi-modal**: Handle images in books

---

## Configuration Examples

### High-Accuracy Configuration
```python
# config.py modifications for better accuracy
config.chunking.chunk_size = 1024  # Larger chunks
config.chunking.chunk_overlap = 100  # More overlap
config.retrieval.top_k = 10  # More context
config.retrieval.similarity_threshold = 0.8  # Stricter threshold
config.generation.temperature = 0.3  # More deterministic
```

### Fast Inference Configuration
```python
# config.py modifications for speed
config.chunking.chunk_size = 256  # Smaller chunks
config.retrieval.top_k = 3  # Less context
config.generation.max_new_tokens = 128  # Shorter answers
config.embedding.batch_size = 64  # Larger batches
```

---

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce batch_size
   - Use smaller chunk_size
   - Process library in smaller batches

2. **Poor Retrieval Quality**:
   - Adjust similarity_threshold
   - Increase chunk_overlap
   - Try different embedding models

3. **Poor Answer Quality**:
   - Increase top_k for more context
   - Improve prompt templates
   - Consider fine-tuning

4. **Slow Indexing**:
   - Increase batch_size
   - Use GPU if available
   - Cache embeddings

---

## Success Metrics

### Indexing Phase
- Successfully parse all EPUBs
- Generate embeddings for all chunks
- Build searchable index

### Query Phase
- Retrieve relevant chunks (manual inspection)
- Generate coherent answers
- Answer accuracy > 80% on test questions

### Performance
- Indexing: < 1 minute per book
- Query latency: < 5 seconds end-to-end
- Index size: Reasonable (< 1GB for 100 books)

---

## Next Steps

1. Start with Phase 1 (Foundation)
2. Validate each phase before moving forward
3. Test with a small library first (2-3 books)
4. Scale up after validation
5. Consider fine-tuning after RAG pipeline is solid

---

## References

- **FAISS**: https://github.com/facebookresearch/faiss
- **Sentence Transformers**: https://www.sbert.net/
- **LangChain**: https://python.langchain.com/
- **Llama 3.2**: https://huggingface.co/meta-llama/Llama-3.2-1B
- **RAG Papers**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
