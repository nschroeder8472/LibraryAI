# LibraryAI

A Retrieval-Augmented Generation (RAG) system for answering questions about your personal ebook library.

## Overview

LibraryAI transforms your collection of EPUB books into an intelligent, searchable knowledge base. Using state-of-the-art embedding models and Meta's Llama 3.2-1B language model, it can:

- Answer questions about content across your entire library
- Recommend books based on topics or themes
- Locate specific passages or concepts
- Provide contextual answers with source citations

## Features

- **EPUB Support**: Automatically parses and indexes EPUB files
- **Semantic Search**: Uses sentence-transformers for high-quality embeddings
- **Fast Retrieval**: FAISS vector store for efficient similarity search
- **Context-Aware Answers**: RAG architecture provides accurate, grounded responses
- **Multiple Query Types**: Q&A, recommendations, and passage location
- **Interactive Mode**: Chat-like interface for exploring your library
- **Source Citations**: Always shows which books and chapters were used

## Architecture

```
EPUB Files → Parser → Text Chunker → Embeddings → FAISS Index
                                                        ↓
User Query → Embedder → Retriever → Context + Prompt → Llama → Answer
```

### Components

- **Data Pipeline**: EPUB parsing and intelligent text chunking
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- **Vector Store**: FAISS with L2 distance
- **LLM**: Meta Llama 3.2-1B for generation
- **Retrieval**: Top-k similarity search with configurable threshold

## Installation

### Prerequisites

- Python 3.9+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install sentence-transformers faiss-cpu langchain beautifulsoup4 lxml ebooklib torch transformers
```

## Docker (GPU)

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose v2](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

### Build

```bash
docker compose build
```

### Usage

```bash
# Show help
docker compose run --rm libraryai --help

# Index your library (place EPUBs in data/raw/ first)
docker compose run --rm libraryai index

# Single query
docker compose run --rm libraryai query "What books discuss AI?"

# Interactive mode
docker compose run --rm libraryai interactive
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `/app/data` | Base data directory |
| `HF_HOME` | `/models` | Hugging Face model cache |
| `HF_TOKEN` | *(none)* | Required for gated models (e.g. Llama) |
| `AUTO_DETECT_DEVICE` | `true` | Auto-detect CUDA/MPS/CPU |
| `EMBEDDING_DEVICE` | auto | Override embedding device |
| `GENERATION_DEVICE` | auto | Override generation device |
| `USE_4BIT` | `false` | Enable 4-bit quantization |
| `USE_8BIT` | `false` | Enable 8-bit quantization |
| `CHUNK_SIZE` | `1000` | Text chunk size in characters |
| `RETRIEVAL_TOP_K` | `5` | Number of retrieved contexts |

Set `HF_TOKEN` in `docker-compose.yml` or pass it at runtime:

```bash
HF_TOKEN=hf_xxx docker compose run --rm libraryai query "your question"
```

## Usage (Local)

### 1. Index Your Library

Place your EPUB files in the `data/raw` directory, then build the index:

```bash
# Using main CLI
python main.py index --library-dir data/raw

# Or using the standalone script
python scripts/build_index.py --library-dir data/raw
```

This will:
- Parse all EPUB files
- Chunk the text into manageable pieces
- Generate embeddings
- Build a FAISS vector index
- Save everything to `data/vector_store/`

### 2. Query Your Library

**Single Query:**

```bash
python main.py query "What books discuss artificial intelligence?"
```

**With source citations:**

```bash
python main.py query "Who are the main characters?" --show-sources
```

**Different query types:**

```bash
# Q&A (default)
python main.py query "What is the main theme of...?" --type qa

# Book recommendations
python main.py query "Suggest books about science fiction" --type recommendation

# Find passages
python main.py query "Where did the author discuss quantum physics?" --type passage_location
```

### 3. Interactive Mode

For a conversational experience:

```bash
python main.py interactive
```

Interactive commands:
- `/type qa` - Switch to Q&A mode
- `/type recommendation` - Switch to recommendation mode
- `/type passage_location` - Switch to passage location mode
- `/sources` - Toggle source display on/off
- `quit` or `exit` - Exit interactive mode

## Configuration

Edit `src/config.py` to customize:

- **Chunking**: chunk size, overlap, separators
- **Embeddings**: model, device (cpu/cuda), batch size
- **Retrieval**: top_k results, similarity threshold
- **Generation**: Llama model, temperature, max tokens

Example:

```python
from src.config import config

# Adjust chunk size for longer context
config.chunking.chunk_size = 1500
config.chunking.chunk_overlap = 300

# Use more retrieved contexts
config.retrieval.top_k = 10

# Make generation more creative
config.generation.temperature = 0.9
```

## Project Structure

```
LibraryAI/
├── main.py                  # Main CLI interface
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── CLAUDE.md              # Project instructions for Claude Code
├── src/
│   ├── config.py          # Configuration management
│   ├── data/
│   │   ├── epub_parser.py # EPUB parsing
│   │   └── chunker.py     # Text chunking
│   ├── embeddings/
│   │   ├── embedder.py    # Embedding generation
│   │   └── vector_store.py# FAISS vector store
│   ├── retrieval/
│   │   └── retriever.py   # Retrieval logic
│   ├── generation/
│   │   ├── model.py       # Llama model wrapper
│   │   └── prompt_templates.py # Prompt engineering
│   ├── pipeline/
│   │   ├── indexing.py    # Indexing orchestration
│   │   └── query.py       # Query orchestration
│   └── utils/
│       └── logging_config.py # Logging setup
├── scripts/
│   ├── build_index.py     # Standalone indexing script
│   └── query_cli.py       # Standalone query script
├── data/
│   ├── raw/              # Your EPUB files (gitignored)
│   ├── processed/        # Parsed books (gitignored)
│   └── vector_store/     # FAISS index (gitignored)
└── plan/
    └── project_plan.md   # Detailed implementation plan
```

## Example Workflow

```bash
# 1. Add EPUB files to data/raw/
cp ~/Books/*.epub data/raw/

# 2. Build the index
python main.py index

# 3. Start querying
python main.py query "What are the key themes across all books?"

# 4. Or use interactive mode
python main.py interactive
```

## Performance Tips

- **Chunk Size**: Smaller chunks (512-1024 chars) = more precise but may lose context
- **Top K**: More results = better context but slower generation
- **Device**: Use CUDA if available for faster embedding/generation
- **Batch Size**: Increase for faster indexing on powerful machines

## Troubleshooting

**Index not found error:**
```
Error: Vector index not found.
```
Solution: Run `python main.py index` first to build the index.

**Out of memory:**
- Reduce chunk size in config
- Reduce batch size for embeddings
- Process fewer books at once

**Slow queries:**
- Reduce `top_k` in retrieval config
- Use smaller max_new_tokens for generation
- Ensure vector store is saved/loaded properly

## Future Enhancements

- Support for PDF and other formats
- Book metadata search (author, title, publication date)
- Reading history tracking
- Multi-modal support (images, tables)
- Fine-tuning Llama on your library
- Web interface
- Advanced query features (filters, boolean search)

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

- **Meta**: Llama 3.2-1B language model
- **Hugging Face**: Transformers library and model hosting
- **Sentence Transformers**: High-quality embedding models
- **FAISS**: Efficient similarity search
- **LangChain**: Text processing utilities
- **ebooklib**: EPUB parsing
