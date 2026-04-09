# LibraryAI v2: Migrate to MemPalace Storage/Retrieval

## Overview

Replace the custom ChromaDB + sentence-transformers + cross-encoder retrieval pipeline with **MemPalace** as the storage and retrieval backend. Keep everything else that already works: Ollama generation, streaming, conversation sessions, web UI, series detection, and EPUB parsing.

### Current state (post-enhancement)

The project already has a mature architecture:
- **Storage**: ChromaDB with hierarchical parent-child chunks (450-char children for search, 1500-char parents for context)
- **Embeddings**: nomic-embed-text-v1.5 via sentence-transformers (768-dim)
- **Retrieval**: ChromaDB vector search → cross-encoder reranking (ms-marco-MiniLM-L-12-v2) → top-8 results
- **Generation**: Ollama with Llama 3.2 (local, streaming via SSE)
- **Features**: Conversation sessions with query rewriting, series detection, chronological retrieval, summary generation, scoped queries (by book/series), library sidebar, dark mode web UI
- **CLI**: `index`, `query`, `interactive`, `web` commands

### Why MemPalace

MemPalace replaces the custom embeddings + ChromaDB + reranker stack with a single system that provides:
- **Hierarchical organization** (Wings/Rooms/Halls) — documented 34% retrieval improvement over flat vector search
- **Progressive context loading** — 170 tokens at startup, deep search on demand
- **Temporal knowledge graph** — track character relationships, plot events with time validity
- **Simpler dependency footprint** — eliminates sentence-transformers, cross-encoder, and manual embedding management
- **Built-in ChromaDB** — MemPalace uses ChromaDB internally, so the underlying tech is familiar

### What changes vs. what stays

| Component | Status | Notes |
|-----------|--------|-------|
| EPUB parsing (`epub_parser.py`) | **Keep** | Works well, no changes needed |
| Series detection (`series_manager.py`) | **Keep** | Maps naturally to MemPalace Wings |
| Hierarchical chunking (`chunker.py`) | **Adapt** | May simplify; MemPalace has its own chunking |
| Embeddings (`embedder.py`) | **Remove** | MemPalace handles embeddings via internal ChromaDB |
| Vector store (`vector_store.py`) | **Replace** | Swap for MemPalace adapter |
| Retriever (`retriever.py`) | **Replace** | Swap for MemPalace `search_memories()` |
| Reranker (`reranker.py`) | **Evaluate** | Wing/Room filtering may replace reranking; keep as optional fallback |
| Ollama generation (`model.py`) | **Keep** | Already works, streaming included |
| Prompt templates (`prompt_templates.py`) | **Keep** | Minor updates for new context format |
| Query pipeline (`query.py`) | **Adapt** | Swap retrieval calls, keep conversation/rewriting logic |
| Indexing pipeline (`indexing.py`) | **Replace** | New ingestion pipeline for MemPalace |
| Summary pipeline (`summary.py`) | **Keep** | Works independently of storage backend |
| Session management (`session.py`) | **Keep** | Storage-agnostic |
| Web app (`app.py`) | **Adapt** | Swap backend calls, keep endpoints/UI |
| Web UI (`index.html`) | **Adapt** | Add wing/room filters, keep existing UI |
| Log buffer (`log_buffer.py`) | **Keep** | No changes |
| Config (`config.py`) | **Adapt** | Replace embedding/retrieval configs with MemPalace config |

### Architecture diagram

```
CURRENT:
  EPUB → Parser → Chunker (parent+child) → Embedder (nomic) → ChromaDB
  Query → ChromaDB search → Reranker (cross-encoder) → Ollama → Answer

NEW:
  EPUB → Parser → SeriesManager → MemPalace ingestion (Wing/Room/Drawer)
  Query → MemPalace search_memories() → [optional reranker] → Ollama → Answer
```

### Deployment target

- Shared homelab server, GPU with 8GB VRAM
- Ollama with Llama 3.2 (~2.5GB VRAM) — already configured
- MemPalace runs locally (ChromaDB + SQLite), no cloud dependencies
- Plex transcodes can run concurrently

---

## Phase 1: Install MemPalace & Explore Its Python API

**Goal:** Install mempalace, understand its programmatic API, and plan the adapter layer.

### Tasks

- [ ] 1.1 Install mempalace: `pip install mempalace`
- [ ] 1.2 Initialize palace: `mempalace init .`
- [ ] 1.3 Read mempalace source to confirm Python API signatures:
  - How to call `add_drawer()` programmatically (vs. MCP tool)
  - How `search_memories()` works (params, return format)
  - How Wings/Rooms are created and managed
  - How the knowledge graph (`kg_add`, `kg_query`) works
  - What metadata is stored per drawer
- [ ] 1.4 Update `requirements.txt`:
  - **Remove**: `sentence-transformers`, `bitsandbytes`
  - **Add**: `mempalace`
  - **Keep everything else** (including `chromadb` if mempalace depends on it, `langchain-text-splitters` if we keep chunking)
- [ ] 1.5 Add `MemPalaceConfig` to `src/config.py`:
  - `palace_path`: path to `.mempalace/` directory
  - `default_wing`: default wing for queries
  - `chunk_size`: drawer chunk size (likely keep ~1500 for prose)
  - Keep `OllamaConfig` / `GenerationConfig` as-is (already working)

### Files to modify
| File | Action |
|------|--------|
| `requirements.txt` | Update dependencies |
| `src/config.py` | Add MemPalaceConfig, remove EmbeddingConfig |

### Verification
- `python -c "import mempalace"` succeeds
- `mempalace status` shows empty palace
- Config loads without errors

---

## Phase 2: MemPalace Ingestion Pipeline

**Goal:** Replace `src/pipeline/indexing.py` with MemPalace-based ingestion that reuses the existing EPUB parser and series detection.

### Design: Book → MemPalace mapping

| Library concept | MemPalace concept | Source |
|----------------|-------------------|--------|
| Series (e.g., "Legacy of the Force") | **Wing** | `SeriesManager` auto-detection |
| Standalone author grouping | **Wing** | `SeriesManager` fallback |
| Individual book | **Room** | Book title from EPUB metadata |
| Chapter text chunks | **Drawers** | Chunked chapter content |
| Character facts, plot events | **Knowledge graph** | Optional: extracted during ingestion |

### Tasks

- [ ] 2.1 Create `src/pipeline/ingest.py`:
  - Reuse `EPUBParser.parse_epub()` for EPUB extraction
  - Reuse `SeriesManager` for series/wing detection
  - For each parsed book:
    - Create/use Wing from series name (or "Author Collection")
    - Create/use Room from book title
    - Chunk chapters (keep ~1500 chars for prose context, or use mempalace's native chunking)
    - Store each chunk as a Drawer with metadata: book_title, book_author, chapter_title, chapter_order, book_order_in_series
  - Deduplicate by title+author hash (library has duplicate EPUBs)
  - Progress logging (reuse existing `logging_config.py`)
- [ ] 2.2 Adapt or simplify `src/data/chunker.py`:
  - If mempalace's native chunking is sufficient, simplify to just chapter-level splitting
  - If not, keep current chunking but remove parent-child complexity (MemPalace's hierarchy replaces it)
- [ ] 2.3 Update `main.py` `index` command to use new `ingest.py` pipeline
- [ ] 2.4 Test: ingest full library, verify with `mempalace status`

### Files to modify
| File | Action |
|------|--------|
| `src/pipeline/ingest.py` | Create (replaces `indexing.py`) |
| `src/pipeline/indexing.py` | Delete after `ingest.py` works |
| `src/data/chunker.py` | Adapt (simplify parent-child if not needed) |
| `src/data/epub_parser.py` | Keep as-is |
| `src/data/series_manager.py` | Keep as-is |
| `main.py` | Update `index` command |

### Verification
- `python main.py index --library-dir data/raw` completes without errors
- `mempalace status` shows correct Wings (per series), Rooms (per book), Drawer counts
- `mempalace search "test character"` returns relevant passages

---

## Phase 3: Search & Retrieval Adapter

**Goal:** Replace `VectorStore` + `Retriever` + `Reranker` with MemPalace search, preserving scoped queries and chronological retrieval.

### Tasks

- [ ] 3.1 Create `src/retrieval/mempalace_retriever.py`:
  - `search(query, top_k, scope)` → calls `search_memories()` with wing/room filtering
  - Scope mapping:
    - `{"type": "series", "name": "X"}` → search within Wing "X"
    - `{"type": "book", "title": "X"}` → search within Room "X"
    - No scope → search entire palace
  - Format results to match existing `RetrievalResult` structure (text, metadata, score) so downstream code (prompt templates, web API) doesn't need major changes
  - `retrieve_chronological(query, scope)` → search + sort by book_order + chapter_order metadata
- [ ] 3.2 Evaluate reranking:
  - Test retrieval quality with MemPalace's wing/room filtering alone
  - If quality is comparable to current cross-encoder reranking, remove `reranker.py`
  - If not, keep reranker as optional post-search step
- [ ] 3.3 Update `src/pipeline/query.py`:
  - Replace `Retriever` instantiation with `MemPalaceRetriever`
  - Keep conversation history, query rewriting, prompt building, and generation logic
  - Keep `generate_stream()` path for SSE
- [ ] 3.4 Delete `src/embeddings/embedder.py` (no longer managing our own embeddings)
- [ ] 3.5 Delete `src/embeddings/vector_store.py` (replaced by MemPalace)
- [ ] 3.6 Test: existing query types (QA, recommendation, passage_location, character_evolution) all work

### Files to modify
| File | Action |
|------|--------|
| `src/retrieval/mempalace_retriever.py` | Create |
| `src/retrieval/retriever.py` | Delete after new retriever works |
| `src/retrieval/reranker.py` | Keep or delete based on quality eval |
| `src/embeddings/embedder.py` | Delete |
| `src/embeddings/vector_store.py` | Delete |
| `src/pipeline/query.py` | Adapt (swap retriever) |

### Verification
- `python main.py query "What happens to Jacen?" --show-sources` returns relevant, cited results
- `python main.py query "..." --type character_evolution` returns chronologically ordered results
- Scoped queries (by book/series) return properly filtered results
- Streaming via web UI still works

---

## Phase 4: Web UI & API Adaptation

**Goal:** Update web backend to use MemPalace, add palace status info. Minimal frontend changes since existing UI is already polished.

### Tasks

- [ ] 4.1 Update `src/web/app.py`:
  - Replace `IndexingPipeline` with new `IngestPipeline` in `/api/index` endpoint
  - Replace retriever initialization with `MemPalaceRetriever`
  - Add `GET /api/palace/status` → wing/room/drawer counts
  - Keep all existing endpoints (query, stream, sessions, library, summaries, logs)
- [ ] 4.2 Update `GET /api/library` endpoint:
  - Pull library structure from MemPalace wings/rooms instead of ChromaDB metadata queries
- [ ] 4.3 Update `src/web/static/index.html` (minor):
  - Show palace status in sidebar or health display
  - Library sidebar should still work (driven by `/api/library` which we're updating)
- [ ] 4.4 Test: full workflow through web UI (ingest, query, stream, sessions)

### Files to modify
| File | Action |
|------|--------|
| `src/web/app.py` | Adapt |
| `src/web/static/index.html` | Minor updates |

### Verification
- Web UI loads, shows library sidebar with series/books
- Ingest via UI works, shows progress
- Chat works with streaming responses
- Session history preserved across queries
- Scoped queries (click a book in sidebar) work correctly

---

## Phase 5: Knowledge Graph Enrichment (Optional)

**Goal:** Leverage MemPalace's temporal knowledge graph for structured queries about characters, events, and relationships.

### Tasks

- [ ] 5.1 Create `src/pipeline/kg_builder.py`:
  - During ingestion, extract structured facts from chapter text
  - Use Ollama to extract: character names, relationships, key events
  - Store via `mempalace kg_add(subject, predicate, object)` with validity windows (book order)
- [ ] 5.2 Add `kg` query type:
  - `python main.py query "Jacen's relationships" --type kg`
  - Queries knowledge graph, not just vector search
- [ ] 5.3 Add `GET /api/kg/query` endpoint for web UI
- [ ] 5.4 Test: temporal queries like "What was true about character X as of book 5?"

### Files to modify
| File | Action |
|------|--------|
| `src/pipeline/kg_builder.py` | Create |
| `src/pipeline/query.py` | Add KG query path |
| `main.py` | Add `kg` query type |
| `src/web/app.py` | Add KG endpoint |

### Verification
- `mempalace kg_query "Jacen Solo"` returns structured facts
- Temporal queries return time-appropriate results

---

## Phase 6: Cleanup & Documentation

**Goal:** Remove obsolete code, update all documentation.

### Files to delete

| File | Reason |
|------|--------|
| `src/embeddings/embedder.py` | MemPalace handles embeddings |
| `src/embeddings/vector_store.py` | Replaced by MemPalace |
| `src/retrieval/retriever.py` | Replaced by `mempalace_retriever.py` |
| `src/retrieval/reranker.py` | If quality eval passes without it |
| `src/pipeline/indexing.py` | Replaced by `ingest.py` |
| `data/vector_store/` | Old ChromaDB data |
| `plan/enhancement_plan.md` | Completed, superseded by this plan |

### Files to update

| File | Change |
|------|--------|
| `CLAUDE.md` | New architecture, commands, dependencies |
| `README.md` | Setup instructions with MemPalace + Ollama |
| `requirements.txt` | Final cleanup of unused dependencies |
| `src/config.py` | Remove dead config classes (EmbeddingConfig, etc.) |
| `scripts/build_index.py` | Update for new ingest pipeline |
| `scripts/query_cli.py` | Update for new retriever |
| Docker files | Update or defer |

### Tasks

- [ ] 6.1 Delete all obsolete files
- [ ] 6.2 Remove empty directories (`src/embeddings/` if empty)
- [ ] 6.3 Update `CLAUDE.md` and `README.md`
- [ ] 6.4 Update Docker files for simplified dependency set
- [ ] 6.5 Final `pip install -r requirements.txt` test on clean env

### Verification
- `pip install -r requirements.txt` on clean virtualenv succeeds
- `python main.py index --library-dir data/raw` → `python main.py query "test"` works end-to-end
- Web UI fully functional
- No import errors from deleted modules

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| MemPalace Python API differs from MCP tool interface | Med | Read source in Phase 1 before writing adapter code |
| Loss of parent-child chunk strategy | Med | Test retrieval quality; if worse, implement similar linking in MemPalace drawers |
| Loss of cross-encoder reranking | Med | Evaluate in Phase 3; keep reranker as optional if Wing/Room filtering isn't enough |
| Existing conversation sessions break | Low | Session manager is storage-agnostic; just need retrieval results in same format |
| Chronological retrieval harder without global_order | Med | Store book_order + chapter_order in drawer metadata; sort in Python after search |
| Scoped queries (by book/series) work differently | Med | Map scope → Wing/Room filter in mempalace_retriever.py |
| MemPalace is relatively new, API may change | Med | Pin version in requirements.txt |
| Summary generation depends on retrieval | Low | Summary pipeline reads chapters directly, not via retrieval |

---

## Model upgrade path (unchanged)

If Llama 3.2 quality is insufficient, upgrade via Ollama config change only:

| Model | VRAM | Quality | Risk with Plex |
|-------|------|---------|----------------|
| Llama 3.2-3B (Q4) | ~2.5GB | Good | None |
| Phi-3 Mini 3.8B (Q4) | ~2.8GB | Good | None |
| Mistral-7B (Q4) | ~4.5GB | Better | Low — tight during 4K transcode |
| Llama 3.1-8B (Q4) | ~5GB | Best | Medium — may compete with 4K transcode |