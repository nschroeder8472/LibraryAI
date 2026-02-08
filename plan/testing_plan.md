# LibraryAI Testing Plan (Phase 8)

## Overview

This document covers the testing strategy for LibraryAI's RAG pipeline. Tests are organized into unit tests (per module), integration tests (cross-module), and end-to-end tests (full pipeline).

All tests use **pytest** and live under `tests/`.

---

## Directory Structure

```
tests/
├── conftest.py                # Shared fixtures
├── test_epub_parser.py        # EPUB parsing tests
├── test_chunker.py            # Text chunking tests
├── test_embedder.py           # Embedding generation tests
├── test_vector_store.py       # FAISS vector store tests
├── test_retriever.py          # Retrieval tests
├── test_prompt_templates.py   # Prompt template tests
├── test_generation.py         # Language model tests
├── test_config.py             # Configuration tests
├── test_indexing_pipeline.py  # Indexing pipeline integration
├── test_query_pipeline.py     # Query pipeline integration
└── test_end_to_end.py         # Full end-to-end tests
```

---

## Shared Fixtures (`conftest.py`)

Common test data and objects reused across test files:

- **`sample_epub_path`** — path to a small test EPUB file (created in `tests/fixtures/`)
- **`sample_book_data`** — pre-parsed book dictionary with title, author, chapters
- **`sample_chapters`** — list of chapter dicts with text content
- **`sample_chunks`** — list of chunk dicts with metadata
- **`sample_embeddings`** — numpy array of embeddings matching `sample_chunks`
- **`embedder`** — `Embedder` instance (loads the sentence-transformer model once per session)
- **`vector_store`** — `VectorStore` populated with `sample_embeddings`
- **`tmp_dir`** — temporary directory for save/load tests (via `tmp_path`)

Use `@pytest.fixture(scope="session")` for expensive fixtures like model loading.

---

## Unit Tests

### 1. EPUB Parser (`test_epub_parser.py`)

| Test | Description |
|------|-------------|
| `test_parse_epub_valid` | Parse a valid EPUB, verify title/author/chapters returned |
| `test_parse_epub_metadata` | Verify all metadata fields (title, author, language, identifier) |
| `test_parse_epub_chapters` | Verify chapters have text, title, item_id, order fields |
| `test_parse_epub_skips_short_chapters` | Chapters with < 50 chars of text are excluded |
| `test_parse_epub_invalid_path` | Raises exception for non-existent file |
| `test_parse_library_finds_nested_epubs` | `parse_library()` finds EPUBs in subdirectories |
| `test_parse_library_empty_dir` | Returns empty list for directory with no EPUBs |
| `test_parse_library_continues_on_error` | Skips unparseable files, parses the rest |
| `test_extract_chapter_title` | Extracts title from h1/h2/h3/title tags |

### 2. Text Chunker (`test_chunker.py`)

| Test | Description |
|------|-------------|
| `test_chunk_text_basic` | Splits text into chunks of correct approximate size |
| `test_chunk_text_overlap` | Consecutive chunks overlap by the configured amount |
| `test_chunk_text_short` | Short text (< chunk_size) produces a single chunk |
| `test_chunk_text_empty` | Empty string returns empty list |
| `test_chunk_chapter_metadata` | Chunk dicts include book_title, book_author, chapter_title, etc. |
| `test_chunk_book` | All chapters chunked; total chunks > number of chapters |
| `test_chunk_library` | Multiple books chunked; chunks from all books present |
| `test_custom_chunk_size` | Custom chunk_size and overlap are respected |

### 3. Embedder (`test_embedder.py`)

| Test | Description |
|------|-------------|
| `test_embed_text_shape` | Single text returns 1D array of correct dimension (384) |
| `test_embed_batch_shape` | Batch of N texts returns (N, 384) array |
| `test_embed_text_deterministic` | Same input produces same embedding |
| `test_embed_chunks_returns_tuple` | Returns (embeddings_array, chunks_with_metadata) |
| `test_embed_chunks_attaches_embedding` | Each chunk dict gets an `embedding` key |
| `test_embed_empty_batch` | Empty list handled gracefully |

### 4. Vector Store (`test_vector_store.py`)

| Test | Description |
|------|-------------|
| `test_add_embeddings` | Adding N embeddings increases `index.ntotal` by N |
| `test_search_returns_top_k` | Search returns exactly `top_k` results |
| `test_search_similarity_scores` | Results include `similarity_score` field |
| `test_search_order` | Results sorted by similarity (ascending L2 distance) |
| `test_save_and_load` | Save to disk, load back; search produces same results |
| `test_get_stats` | Stats dict has total_vectors, embedding_dim, total_chunks |
| `test_add_single_embedding` | 1D embedding is reshaped and added correctly |
| `test_search_empty_index` | Searching an empty index returns empty results |

### 5. Retriever (`test_retriever.py`)

| Test | Description |
|------|-------------|
| `test_retrieve_returns_chunks` | Retrieve returns list of chunk dicts |
| `test_retrieve_respects_top_k` | Number of results <= top_k |
| `test_retrieve_threshold_filtering` | Chunks above threshold are filtered out |
| `test_format_context_nonempty` | Formats chunks into readable context string |
| `test_format_context_empty` | Returns "No relevant information" message |
| `test_format_context_includes_metadata` | Context string contains book title, author, chapter |

### 6. Prompt Templates (`test_prompt_templates.py`)

| Test | Description |
|------|-------------|
| `test_qa_prompt_contains_context` | QA prompt includes the provided context |
| `test_qa_prompt_contains_query` | QA prompt includes the user query |
| `test_recommendation_prompt_with_history` | Recommendation prompt includes reading history |
| `test_recommendation_prompt_without_history` | Works without reading history |
| `test_passage_location_prompt` | Passage location prompt includes context and query |

### 7. Language Model (`test_generation.py`)

| Test | Description |
|------|-------------|
| `test_model_loads` | Model and tokenizer load without error |
| `test_generate_returns_string` | `generate()` returns a non-empty string |
| `test_generate_respects_max_tokens` | Output length bounded by max_new_tokens |
| `test_pad_token_set` | Tokenizer has pad_token set |

> **Note**: LLM tests are slow and require model download. Mark with `@pytest.mark.slow` and skip by default in CI.

### 8. Configuration (`test_config.py`)

| Test | Description |
|------|-------------|
| `test_default_config` | Default config loads without error |
| `test_env_override_data_dir` | DATA_DIR env var overrides data directory |
| `test_env_override_embedding_device` | EMBEDDING_DEVICE env var respected |
| `test_env_override_generation_device` | GENERATION_DEVICE env var respected |
| `test_directories_created` | Config creates data directories on init |
| `test_auto_detect_device` | AUTO_DETECT_DEVICE selects cuda/mps/cpu appropriately |

---

## Integration Tests

### Indexing Pipeline (`test_indexing_pipeline.py`)

| Test | Description |
|------|-------------|
| `test_index_library_end_to_end` | Parse -> chunk -> embed -> store for a test EPUB |
| `test_index_saves_to_disk` | After indexing, vector store files exist on disk |
| `test_index_parsed_data_saved` | parsed_books.json written to processed dir |
| `test_index_stats_correct` | Stats reflect actual number of chunks indexed |

### Query Pipeline (`test_query_pipeline.py`)

| Test | Description |
|------|-------------|
| `test_query_returns_answer` | Query returns dict with 'answer' key |
| `test_query_returns_contexts` | Query returns retrieved context chunks |
| `test_query_types` | qa, recommendation, passage_location all work |
| `test_query_no_results` | Graceful response when no relevant chunks found |

---

## End-to-End Tests (`test_end_to_end.py`)

| Test | Description |
|------|-------------|
| `test_index_then_query` | Index a test EPUB, then query it successfully |
| `test_index_then_interactive_query` | Index, then simulate an interactive session |
| `test_multiple_books` | Index multiple EPUBs, query across them |
| `test_cli_help` | `main.py --help` exits cleanly |
| `test_cli_index_missing_dir` | `main.py index --library-dir /nonexistent` errors gracefully |

> **Note**: End-to-end tests require the LLM model. Mark with `@pytest.mark.slow`.

---

## Test Markers

```ini
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests that require model downloads or heavy computation",
    "integration: marks integration tests",
    "e2e: marks end-to-end tests",
]
```

---

## Running Tests

```bash
# Run all fast tests (excludes slow/model-dependent tests)
pytest tests/ -m "not slow"

# Run all tests including slow ones
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_chunker.py -v

# Run in Docker (local)
docker compose -f docker-compose.local.yml run --rm test
```

---

## Test Data

### Minimal test EPUB

Create a small EPUB fixture (`tests/fixtures/test_book.epub`) with:
- 2-3 short chapters (100-200 words each)
- Known title and author metadata
- Deterministic content for assertion matching

Alternatively, build the EPUB programmatically in a `conftest.py` fixture using `ebooklib`.

### Sample chunk data

Pre-built chunk dicts for tests that don't need a real EPUB:
```python
SAMPLE_CHUNKS = [
    {
        "text": "The quick brown fox jumps over the lazy dog.",
        "book_title": "Test Book",
        "book_author": "Test Author",
        "chapter_title": "Chapter 1",
        "chapter_order": 0,
        "chunk_index": 0,
        "total_chunks_in_chapter": 1,
    },
    # ... more chunks
]
```
