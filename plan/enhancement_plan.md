# LibraryAI Enhancement Plan

This plan addresses seven feature requests and includes additional improvement recommendations. Each section analyzes the current state, proposes specific changes, and identifies the files affected.

---

## Table of Contents

1. [Improve Answer Accuracy (Fine Details & Character Evolution)](#1-improve-answer-accuracy)
2. [Book and Series Summaries](#2-book-and-series-summaries)
3. [Chronological Awareness & Character Tracking](#3-chronological-awareness--character-tracking)
4. [UI: Book/Series Browser with Scoped Queries](#4-ui-bookseries-browser-with-scoped-queries)
5. [Conversational Tone & Conversation Memory](#5-conversational-tone--conversation-memory)
6. [UI: New Chat Button](#6-ui-new-chat-button)
7. [UI: Logs Page](#7-ui-logs-page)
8. [Additional Recommendations](#8-additional-recommendations)
9. [Implementation Order & Dependencies](#9-implementation-order--dependencies)

---

## 1. Improve Answer Accuracy

### Problem

The current pipeline retrieves 1500-character chunks and feeds them to the LLM. Fine character details (e.g., a character's eye color changing after an event in Book 3) are easily missed because:

- **Chunks are too coarse**: 1500 chars with 300 overlap may split a key detail across chunk boundaries or bury it in surrounding narrative.
- **Retrieval recalls only ~5 chunks**: After reranking, the LLM sees roughly 5 passages. If the relevant detail didn't rank in the top 5, it's invisible.
- **No entity-aware indexing**: Chunks don't carry structured annotations about *which characters or events* they reference, so a query about "Rand's hand" must rely purely on keyword/semantic overlap.

### Proposed Changes

#### A. Smaller, Overlapping Chunks with a Parent-Child Strategy

Use a **two-tier chunking** approach (sometimes called "small-to-big"):

- **Child chunks** (400-500 chars): Used for embedding and retrieval. Smaller chunks produce more precise embeddings because less irrelevant text dilutes the signal.
- **Parent chunks** (1500-2000 chars): The surrounding context window. When a child chunk matches, the *parent* chunk is what gets sent to the LLM, giving it enough narrative context to answer well.

**Files affected**: `src/data/chunker.py`, `src/embeddings/vector_store.py`, `src/retrieval/retriever.py`

**Implementation**:
1. `TextChunker` gets a new method `chunk_chapter_hierarchical()` that produces both child and parent chunks, linking children to parents via a `parent_id` field in metadata.
2. `VectorStore` indexes *child* embeddings but stores *parent* metadata alongside them. On retrieval, it returns the parent text.
3. Deduplicate parents: if multiple child chunks from the same parent match, only include the parent once.

#### B. Increase Retrieval Depth Before Reranking

- Increase `RETRIEVAL_TOP_K` from 10 to 20-30 (the initial FAISS search is cheap).
- Keep `RERANK_TOP_N` at 5-8 to control LLM context length.
- This gives the reranker a wider pool to find the truly relevant passages.

**Files affected**: `src/config.py` (default values), `src/retrieval/retriever.py`

#### C. Upgrade the Embedding Model

The current `BAAI/bge-base-en-v1.5` (768-dim) is solid but there are better options now:

| Model | Dimensions | MTEB Score | Notes |
|-------|-----------|------------|-------|
| `BAAI/bge-base-en-v1.5` (current) | 768 | ~63 | Good baseline |
| **`BAAI/bge-large-en-v1.5`** | 1024 | ~64 | Better accuracy, ~2x memory |
| **`nomic-ai/nomic-embed-text-v1.5`** | 768 | ~65 | Matryoshka dims, good quality |
| **`Alibaba-NLP/gte-large-en-v1.5`** | 1024 | ~66 | Top-tier open-source |

**Recommendation**: Switch to `nomic-ai/nomic-embed-text-v1.5` — it matches or beats `bge-large` while keeping 768 dims (less memory, faster search).

**Files affected**: `src/config.py` (default model name), documentation

#### D. Upgrade the Reranker

The current `cross-encoder/ms-marco-MiniLM-L-6-v2` was trained on web search data. Consider:

- **`BAAI/bge-reranker-v2-m3`**: Better at long-document, narrative-style reranking.
- **`cross-encoder/ms-marco-MiniLM-L-12-v2`**: Deeper model, better accuracy, still fast.

**Files affected**: `src/config.py`

#### E. Switch to a Better LLM (or Use an API)

`Mistral-7B-Instruct-v0.3` is reasonable but for higher accuracy, consider:

| Option | Pros | Cons |
|--------|------|------|
| **`mistralai/Mistral-Small-24B-Instruct-2501`** | Much better comprehension, still local | Needs ~16GB VRAM (4-bit) |
| **OpenAI `gpt-4o-mini` API** | Excellent comprehension, cheap ($0.15/1M input) | Requires API key, data leaves local |
| **Anthropic Claude API (`claude-sonnet-4-20250514`)** | Top-tier reasoning, great at citing sources | API cost, data leaves local |
| **`Qwen/Qwen2.5-7B-Instruct`** | Strong 7B model, Apache license | Similar VRAM to Mistral |
| **Ollama + local model** | Easy to run locally, many model options | Needs Ollama installed |

**Recommendation**: Add an **API provider abstraction** so users can choose between local models and API-based models (OpenAI, Anthropic, Ollama). This is the single highest-impact change for answer quality. A cloud API model will dramatically outperform any 7B local model at understanding nuanced character details.

**Files affected**: `src/generation/model.py` (add `APILanguageModel` class), `src/config.py` (add `GENERATION_BACKEND` setting: `local` | `openai` | `anthropic` | `ollama`)

---

## 2. Book and Series Summaries

### Problem

There is no summary generation capability. The system is purely query-response.

### Proposed Changes

#### A. Series Metadata System

Add a **series configuration file** (`data/series.json` or UI-managed) that defines:

```json
{
  "series": [
    {
      "name": "The Wheel of Time",
      "author": "Robert Jordan",
      "books": [
        { "title": "The Eye of the World", "order": 1 },
        { "title": "The Great Hunt", "order": 2 }
      ]
    }
  ]
}
```

**Auto-detection**: During indexing, group books by author and attempt to detect series from EPUB metadata (the `calibre:series` and `calibre:series_index` metadata fields are common in Calibre-managed libraries). Provide a UI for manual editing.

**Files affected**: New file `src/data/series_manager.py`, `src/pipeline/indexing.py`, `src/web/app.py` (new API endpoints)

#### B. Summary Generation Pipeline

Add a new `SummaryPipeline` that generates summaries by:

1. **Book summary**: Retrieve all chunks for a specific book (filter metadata), feed them to the LLM in chapter order with a summarization prompt. For long books, use a **map-reduce** approach:
   - **Map**: Summarize each chapter independently (chunked if needed).
   - **Reduce**: Combine chapter summaries into a book summary.

2. **Series summary**: Summarize each book first, then combine book summaries in series order.

3. **Caching**: Store generated summaries in `data/summaries/` so they don't need to be regenerated.

**Files affected**: New `src/pipeline/summary.py`, new prompt templates in `src/generation/prompt_templates.py`, `src/web/app.py` (new `/api/summary` endpoint)

#### C. New Prompt Templates

```python
@staticmethod
def book_summary_prompt(book_title: str, chapter_summaries: str) -> str:
    """Generate a comprehensive book summary."""

@staticmethod
def chapter_summary_prompt(book_title: str, chapter_title: str, chapter_text: str) -> str:
    """Summarize a single chapter."""

@staticmethod
def series_summary_prompt(series_name: str, book_summaries: str) -> str:
    """Generate a series overview."""
```

---

## 3. Chronological Awareness & Character Tracking

### Problem

The current system treats all chunks as a flat bag of text. It has no concept of:
- Book reading order within a series
- Character evolution across books
- Timeline of events

### Proposed Changes

#### A. Enriched Chunk Metadata

During indexing, add to every chunk:

```python
{
    # Existing fields...
    "series_name": "The Wheel of Time",
    "book_order_in_series": 1,
    "chapter_order": 3,
    "global_order": 103,  # Unique ordering across entire series
}
```

The `global_order` field allows the retriever to sort results chronologically.

**Files affected**: `src/data/chunker.py`, `src/data/series_manager.py`

#### B. Entity Extraction During Indexing (Character Index)

Run a lightweight NER (Named Entity Recognition) pass during indexing to extract character names and associate them with chunks:

- Use `spaCy` (fast, local) or the LLM itself for entity extraction.
- Build a **character index**: a mapping from character name → list of chunk IDs where they appear.
- Store character metadata: aliases, first appearance book/chapter.

This allows the query pipeline to:
1. Detect character names in the user's question.
2. Preferentially retrieve chunks that mention that character.
3. Sort those chunks chronologically to show character evolution.

**Files affected**: New `src/data/entity_extractor.py`, `src/embeddings/vector_store.py` (add metadata filtering), `src/retrieval/retriever.py` (add entity-aware retrieval)

#### C. Chronological Context Formatting

When the user asks about a character's evolution, the retriever:
1. Identifies the character from the query.
2. Retrieves chunks mentioning that character.
3. Sorts by `global_order` (book order → chapter order → chunk order).
4. Formats context with clear chronological markers.

New prompt template:

```python
@staticmethod
def character_evolution_prompt(query: str, character: str, chronological_context: str) -> str:
    """Answer questions about character changes across books."""
```

**Files affected**: `src/retrieval/retriever.py` (new `retrieve_character_timeline` method), `src/generation/prompt_templates.py`, `src/pipeline/query.py`

#### D. Metadata Filtering in the Vector Store

FAISS itself doesn't support metadata filtering. Two options:

1. **Post-filter** (simpler): Retrieve a large set (top 50-100) from FAISS, then filter by book/series/character in Python. This works well for moderate library sizes.

2. **Switch to a metadata-aware vector store** (more robust): Replace FAISS with **ChromaDB** or **Qdrant** (both support metadata filtering natively, both have Python packages, both can run embedded/local).

**Recommendation**: Switch to **ChromaDB**. It's a drop-in replacement that adds metadata filtering, is pure Python, runs embedded (no server needed), and is well-maintained. This simplifies filtering by book, series, character, and chapter.

**Files affected**: `src/embeddings/vector_store.py` (rewrite to use ChromaDB), `requirements.txt`

---

## 4. UI: Book/Series Browser with Scoped Queries

### Problem

The UI currently has no awareness of what books exist in the index. Users can't scope their questions to a specific book or series.

### Proposed Changes

#### A. New API Endpoints

```
GET  /api/library              → List all books and series
GET  /api/library/books        → List books with metadata
GET  /api/library/series       → List series with books
POST /api/query                → Add optional "scope" parameter
```

The query endpoint gets new optional fields:

```json
{
  "query": "What color are Rand's eyes?",
  "query_type": "qa",
  "scope": {
    "type": "book",
    "title": "The Eye of the World"
  }
}
```

**Files affected**: `src/web/app.py` (new endpoints, modify query endpoint)

#### B. Backend: Scoped Retrieval

When scope is provided, filter retrieved chunks to only include those from the specified book or series. With ChromaDB (recommended in section 3D), this is a native `where` filter. With FAISS, it's a post-retrieval filter.

**Files affected**: `src/retrieval/retriever.py` (add `scope` parameter to `retrieve()`), `src/pipeline/query.py`

#### C. UI: Sidebar with Library Browser

Redesign the UI layout to include a **collapsible left sidebar**:

```
┌──────────────────────────────────────────────────────────┐
│ Header: LibraryAI    [Query Type ▾] [New Chat] [Logs]    │
├───────────┬──────────────────────────────────────────────┤
│ Library   │  Chat Area                                   │
│           │                                              │
│ [All]     │  [user message]                              │
│           │         [assistant response]                 │
│ ▾ Series  │                                              │
│   Book 1  │  [user message]                              │
│   Book 2  │         [assistant response]                 │
│ ▾ Series  │                                              │
│   Book 3  │                                              │
│           │                                              │
│ Ungrouped │                                              │
│   Book 4  │                                              │
├───────────┴──────────────────────────────────────────────┤
│ [Scope: The Eye of the World ✕]  [Query Input]   [Send]  │
└──────────────────────────────────────────────────────────┘
```

- Clicking a book/series sets the query scope.
- A "Full Library" option at the top removes scoping.
- Active scope is displayed as a badge/chip above or beside the chat input.
- The sidebar is collapsible on mobile.

**Files affected**: `src/web/static/index.html` (major UI rework)

**Recommendation**: Consider splitting the frontend from a single HTML file into separate files (`index.html`, `app.js`, `styles.css`) to manage the growing complexity. A lightweight reactive library like **Alpine.js** (14KB, adds directly via `<script>` tag, no build step) could simplify state management without a full framework migration.

---

## 5. Conversational Tone & Conversation Memory

### Problem

- Each query is independent — no conversation history.
- The prompt is instructed to be "precise" and "direct" — not conversational.
- The web UI sends one query at a time with no session context.

### Proposed Changes

#### A. Server-Side Conversation Sessions

Add a session system to the backend:

```python
# src/web/session.py
class ConversationSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: list[dict] = []  # [{"role": "user", "content": "..."}, ...]
        self.scope: dict | None = None  # Current book/series filter
        self.created_at: datetime
        self.last_activity: datetime

class SessionManager:
    def __init__(self, max_history: int = 20):
        self.sessions: dict[str, ConversationSession] = {}

    def create_session(self) -> str: ...
    def get_session(self, session_id: str) -> ConversationSession: ...
    def add_message(self, session_id: str, role: str, content: str): ...
    def get_history_for_prompt(self, session_id: str, max_turns: int = 5) -> str: ...
    def clear_session(self, session_id: str): ...
```

**Files affected**: New `src/web/session.py`, `src/web/app.py` (modify query endpoint)

#### B. Conversation-Aware Prompting

Include recent conversation history in the LLM prompt:

```python
@staticmethod
def conversational_qa_prompt(query: str, context: str, conversation_history: str) -> str:
    prompt = f"""You are a knowledgeable and friendly librarian assistant who helps users
explore their personal ebook library. You're conversational but accurate.

Previous conversation:
{conversation_history}

Relevant passages from the library:
{context}

User: {query}

Instructions:
- Answer based on the provided passages. If you reference something from the conversation
  history, still ground it in the library passages.
- Be conversational and natural. Use the conversation context to understand follow-up questions.
- If the user says "that character" or "the book you mentioned", resolve the reference from
  the conversation history.
- Cite the book and chapter when providing specific details.
- If you don't have enough information, say so warmly rather than bluntly.

Answer:"""
```

**Files affected**: `src/generation/prompt_templates.py` (update all templates to accept optional `conversation_history`)

#### C. Query Rewriting with Conversation Context

A key problem with follow-up questions is that they're often ambiguous without context. For example:
- User: "Tell me about Rand al'Thor"
- User: "What about his relationship with Egwene?"

The second query alone doesn't mention Rand. **Query rewriting** uses the conversation history to expand the query before retrieval:

- Take the last N turns of conversation + current query.
- Use the LLM (or a fast smaller model) to rewrite the query as a standalone question: "What is the relationship between Rand al'Thor and Egwene al'Vere?"
- Use the rewritten query for embedding and retrieval.

This can be done with a simple prompt:

```python
@staticmethod
def query_rewrite_prompt(conversation_history: str, current_query: str) -> str:
    return f"""Given the conversation history, rewrite the user's latest message as a
standalone question that captures the full intent.

Conversation:
{conversation_history}

Latest message: {current_query}

Rewritten standalone question:"""
```

**Files affected**: `src/pipeline/query.py` (add rewrite step before retrieval), `src/generation/prompt_templates.py`

#### D. Conversation History Window Management

To avoid exceeding LLM context limits:
- Keep the last 5 turns (10 messages) in the prompt by default.
- For longer conversations, summarize older turns.
- Track total token count and truncate if approaching model limits.

**Files affected**: `src/web/session.py`, `src/config.py` (add `MAX_CONVERSATION_TURNS` setting)

---

## 6. UI: New Chat Button

### Problem

No way to clear the current conversation and start fresh.

### Proposed Changes

#### A. Backend: Session Reset Endpoint

```
POST /api/session/new  → Creates new session, returns session_id
```

This clears the conversation history and any active scope.

**Files affected**: `src/web/app.py`

#### B. Frontend: New Chat Button

Add a "New Chat" button in the header bar, next to the query type selector.

Behavior:
1. Calls `POST /api/session/new` to get a fresh session.
2. Clears the chat DOM.
3. Resets the scope selector to "Full Library".
4. Displays the welcome message again.
5. Focuses the input field.

```javascript
async function startNewChat() {
    const res = await fetch("/api/session/new", { method: "POST" });
    const data = await res.json();
    sessionId = data.session_id;
    chat.innerHTML = "";
    currentScope = null;
    updateScopeDisplay();
    addMessage("assistant", "New conversation started. Ask me anything about your library!");
    input.focus();
}
```

**Files affected**: `src/web/static/index.html` (add button + handler)

This is a straightforward change — low effort, high usability impact.

---

## 7. UI: Logs Page

### Problem

Logs are only available in the server console. Users running the Docker container or web server can't easily see what's happening.

### Proposed Changes

#### A. Backend: In-Memory Log Buffer

Create a log handler that captures recent log messages in a circular buffer:

```python
# src/utils/log_buffer.py
import logging
from collections import deque
from datetime import datetime

class LogBuffer(logging.Handler):
    def __init__(self, max_entries: int = 500):
        super().__init__()
        self.buffer: deque = deque(maxlen=max_entries)

    def emit(self, record):
        entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": self.format(record),
        }
        self.buffer.append(entry)

    def get_logs(self, level: str = None, limit: int = 100) -> list[dict]:
        logs = list(self.buffer)
        if level:
            logs = [l for l in logs if l["level"] == level.upper()]
        return logs[-limit:]
```

**Files affected**: New `src/utils/log_buffer.py`, `src/utils/logging_config.py` (attach buffer handler)

#### B. Backend: Log API Endpoints

```
GET /api/logs                    → Get recent logs
GET /api/logs?level=ERROR        → Filter by level
GET /api/logs?limit=50           → Limit results
GET /api/logs/stream             → SSE stream for real-time logs (optional)
```

**Files affected**: `src/web/app.py`

#### C. Frontend: Logs Page

Add a logs view accessible via a "Logs" button in the header. Two approaches:

**Option 1: Separate page** (simpler)
- Navigate to `/logs` which serves a dedicated logs HTML page.
- Auto-refresh every few seconds or use SSE for real-time updates.

**Option 2: Slide-over panel** (better UX)
- A panel slides in from the right when clicking "Logs".
- Shows logs in a scrollable, monospace-font container.
- Filter buttons for log levels (DEBUG, INFO, WARNING, ERROR).
- Auto-scrolls to bottom for new logs.
- "Clear" button to clear the display (not the buffer).

**Recommendation**: Option 2 (slide-over panel). It keeps the user in context and doesn't require page navigation.

```
┌──────────────────────────────────────────────────┐
│ Header: LibraryAI                    [Logs ✕]    │
├──────────────────────┬───────────────────────────┤
│  Chat Area           │  Logs Panel               │
│                      │  [ALL] [INFO] [WARN] [ERR]│
│  ...                 │                           │
│                      │  10:31:02 INFO  Loading...│
│                      │  10:31:05 INFO  Indexed.. │
│                      │  10:31:08 WARN  Low sim.. │
│                      │  10:32:01 INFO  Query...  │
│                      │  10:32:03 INFO  Retrieved │
│                      │  10:32:05 INFO  Generated │
├──────────────────────┴───────────────────────────┤
│ [Query Input]                           [Send]   │
└──────────────────────────────────────────────────┘
```

**Files affected**: `src/web/static/index.html` (add logs panel + button), `src/web/app.py` (add logs endpoint)

---

## 8. Additional Recommendations

These are ideas beyond the seven requested features that would improve the overall experience.

### A. Streaming Responses

Currently the UI waits for the full LLM response before displaying it. **Streaming** the response token-by-token creates a much more responsive feel.

**Implementation**:
- Use **Server-Sent Events (SSE)** or **WebSockets** for the query endpoint.
- The Hugging Face `TextIteratorStreamer` class supports token-by-token streaming from local models.
- API providers (OpenAI, Anthropic) support streaming natively.
- The frontend renders tokens as they arrive.

**Files affected**: `src/generation/model.py`, `src/web/app.py`, `src/web/static/index.html`

### B. Hybrid Search (Semantic + Keyword)

Pure semantic search can miss exact keyword matches (e.g., a character's specific name spelling, a place name). Add **BM25 keyword search** alongside the vector search and fuse the results:

- Use `rank_bm25` Python package for keyword scoring.
- Retrieve top-K from both vector search and BM25.
- Fuse with **Reciprocal Rank Fusion (RRF)** to combine rankings.
- Feed the fused top-N to the reranker.

This is especially useful for questions like "What happened at Dumai's Wells?" where the place name is critical.

**Files affected**: New `src/retrieval/keyword_search.py`, `src/retrieval/retriever.py` (add fusion logic)

### C. Markdown Rendering in the UI

Currently the LLM output is displayed as plain text (with `escapeHtml`). LLM responses often include markdown formatting (bold, lists, headers). Rendering markdown would make responses more readable.

- Use a lightweight markdown library like **marked.js** (~30KB) or **markdown-it** (~50KB).
- Sanitize output with DOMPurify to prevent XSS.
- Add CSS for rendered markdown elements.

**Files affected**: `src/web/static/index.html`

### D. Export/Share Conversations

Allow users to export a conversation as a text file or markdown document.

**Files affected**: `src/web/static/index.html` (add export button), `src/web/app.py` (optional server-side export endpoint)

### E. Dark Mode

A simple CSS variable toggle for dark theme. Low effort, nice quality-of-life improvement.

**Files affected**: `src/web/static/index.html` (CSS variables + toggle button)

### F. Progress Indicators for Long Operations

Summary generation and re-indexing can be slow. Add progress indicators:
- Chunked progress for summary generation (e.g., "Summarizing chapter 3/12...").
- Use SSE to stream progress updates to the UI.

**Files affected**: `src/web/app.py`, `src/web/static/index.html`

---

## 9. Implementation Order & Dependencies

The features have natural dependencies. Here is the recommended implementation order, grouped into phases:

### Phase 1: Foundation (enables everything else)

| Step | Feature | Effort | Rationale |
|------|---------|--------|-----------|
| 1.1 | **Series metadata system** (2A) | Medium | Needed for chronological ordering, scoped queries, and summaries |
| 1.2 | **Enriched chunk metadata** (3A) | Low | Adds series/order info to chunks during indexing |
| 1.3 | **Switch to ChromaDB** (3D) | Medium | Enables metadata filtering needed by scoping, character tracking, and summaries |
| 1.4 | **Hierarchical chunking** (1A) | Medium | Improves retrieval precision for all downstream features |

After Phase 1, re-index the library to pick up new metadata and chunk structure.

### Phase 2: Core Intelligence

| Step | Feature | Effort | Rationale |
|------|---------|--------|-----------|
| 2.1 | **LLM provider abstraction** (1E) | Medium | Higher quality LLM improves all features (accuracy, summaries, conversation) |
| 2.2 | **Upgrade embedding model** (1C) | Low | Config change + re-index |
| 2.3 | **Upgrade reranker** (1D) | Low | Config change only |
| 2.4 | **Increase retrieval depth** (1B) | Low | Config change only |
| 2.5 | **Conversation sessions** (5A) | Medium | Backend infrastructure for memory |
| 2.6 | **Conversational prompting + query rewriting** (5B, 5C) | Medium | Depends on sessions |

### Phase 3: UI Enhancements

| Step | Feature | Effort | Rationale |
|------|---------|--------|-----------|
| 3.1 | **New Chat button** (6) | Low | Quick win, depends on sessions from 2.5 |
| 3.2 | **Library browser sidebar** (4C) | High | Major UI rework |
| 3.3 | **Scoped queries** (4A, 4B) | Medium | Connects sidebar to backend |
| 3.4 | **Logs page/panel** (7) | Medium | Independent, can be done in parallel |

### Phase 4: Advanced Features

| Step | Feature | Effort | Rationale |
|------|---------|--------|-----------|
| 4.1 | **Summary generation pipeline** (2B, 2C) | High | Depends on series metadata, scoped retrieval |
| 4.2 | **Entity extraction / character index** (3B) | High | NER pipeline during indexing |
| 4.3 | **Chronological retrieval** (3C) | Medium | Depends on entity extraction |
| 4.4 | **Hybrid search** (8B) | Medium | Independent improvement |

### Phase 5: Polish

| Step | Feature | Effort | Rationale |
|------|---------|--------|-----------|
| 5.1 | **Streaming responses** (8A) | Medium | UX improvement |
| 5.2 | **Markdown rendering** (8C) | Low | Quick UX win |
| 5.3 | **Dark mode** (8E) | Low | Quick UX win |
| 5.4 | **Export conversations** (8D) | Low | Nice-to-have |

### Dependency Graph

```
Series Metadata (1.1) ──┬──→ Enriched Chunks (1.2) ──→ ChromaDB (1.3)
                        │                                    │
                        ├──→ Summary Pipeline (4.1)          │
                        │                                    ▼
                        └──→ Library Browser UI (3.2) ──→ Scoped Queries (3.3)

LLM Abstraction (2.1) ──→ Better Answers, Better Summaries, Better Conversation

Sessions (2.5) ──┬──→ Conversational Prompts (2.6)
                 └──→ New Chat Button (3.1)

Entity Extraction (4.2) ──→ Chronological Retrieval (4.3)

Hierarchical Chunking (1.4) ──→ (independent, improves all retrieval)
Logs Panel (3.4) ──→ (independent)
Streaming (5.1) ──→ (independent)
```

---

## Summary of New Files

| File | Purpose |
|------|---------|
| `src/data/series_manager.py` | Series detection, ordering, metadata management |
| `src/data/entity_extractor.py` | Character/entity NER extraction during indexing |
| `src/pipeline/summary.py` | Map-reduce summary generation for books and series |
| `src/web/session.py` | Conversation session management |
| `src/utils/log_buffer.py` | In-memory log buffer for the logs UI |
| `src/retrieval/keyword_search.py` | BM25 keyword search for hybrid retrieval |

## Summary of Modified Files

| File | Changes |
|------|---------|
| `src/config.py` | New settings for sessions, LLM backend, retrieval depth, series config |
| `src/data/chunker.py` | Add hierarchical chunking method |
| `src/embeddings/vector_store.py` | Replace FAISS with ChromaDB, add metadata filtering |
| `src/retrieval/retriever.py` | Add scoped retrieval, entity-aware retrieval, hybrid search fusion |
| `src/generation/model.py` | Add API provider abstraction (OpenAI, Anthropic, Ollama) |
| `src/generation/prompt_templates.py` | New templates for conversations, summaries, character evolution |
| `src/pipeline/indexing.py` | Integrate series detection, entity extraction, hierarchical chunking |
| `src/pipeline/query.py` | Add session handling, query rewriting, scoped retrieval |
| `src/web/app.py` | New endpoints for library, sessions, logs, summaries |
| `src/web/static/index.html` | Sidebar, new chat button, logs panel, scope display, markdown rendering |
| `requirements.txt` | Add chromadb, spacy, rank-bm25, marked.js, etc. |