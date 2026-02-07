# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LibraryAI is a fine-tuned AI model for answering questions about a personal ebook library. It uses Meta's Llama 3.2-1B as the base model with Hugging Face transformers for fine-tuning.

## Commands

```bash
# Install dependencies (local)
pip install torch>=2.0.0
pip install -r requirements.txt

# Run the main script
python main.py

# Docker build & run
docker compose build
docker compose run --rm libraryai --help
docker compose run --rm libraryai index
docker compose run --rm libraryai query "your question"
docker compose run --rm libraryai interactive
```

## Architecture

The project follows the standard Hugging Face fine-tuning workflow:
- **Model**: Llama 3.2-1B (meta-llama/Llama-3.2-1B) - a lightweight 1B parameter model suitable for fine-tuning
- **Framework**: Hugging Face transformers with the Trainer API for fine-tuning
- **Data handling**: Hugging Face datasets library with custom tokenization

Current implementation status: Initial skeleton with model/tokenizer loading. Training loop and inference interface are not yet implemented.