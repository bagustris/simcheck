# simcheck

*Detect semantic similarity between student PDF reports using local LLM embeddings*

[![Python](https://img.shields.io/badge/Python->=3.12-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

[Features](#features) • [Getting started](#getting-started) • [Usage](#usage) • [How it works](#how-it-works) • [Development](#development)

---

`simcheck` is a command-line tool that computes pairwise cosine similarity between PDF documents using local GGUF embeddings via [llama.cpp](https://github.com/ggerganov/llama.cpp). It outputs a similarity matrix as CSV and a colour-coded heatmap PDF, and can flag the most similar document pairs.

Everything runs **entirely offline** — no external APIs, no cloud services, no GPU required.

## Features

- **Local-only** — all computation runs on your machine using a GGUF embedding model
- **PDF text extraction** — extracts and normalises text from multi-page PDFs via [PyMuPDF](https://pymupdf.readthedocs.io/)
- **Embedding cache** — embeddings are cached to `.npz` files; re-runs skip already-processed documents
- **Heatmap output** — generates a visual similarity heatmap as a PDF alongside the raw CSV matrix
- **Top-k flagging** — optionally prints the most similar document pairs to stdout
- **Cross-set comparison** — compare current reports against a previous year's set with `--additional_dir`

## Getting started

### Prerequisites

- Python 3.12+
- A GGUF embedding model file (e.g. from [Hugging Face](https://huggingface.co/models?search=gguf+embedding))

### Installation

```bash
pip install .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

> [!NOTE]
> Building `llama-cpp-python` requires a C compiler. On most systems this is already available. See the [llama-cpp-python docs](https://github.com/abetlen/llama-cpp-python#installation) if you run into build issues.

## Usage

```bash
simcheck \
  --input_dir ./reports \
  --model_path ./models/embedding.gguf \
  --output_dir ./output \
  --top_k 5
```

### CLI options

| Option | Required | Description |
|---|---|---|
| `--input_dir PATH` | Yes | Folder containing student PDF files |
| `--model_path PATH` | Yes | Path to a GGUF embedding model |
| `--output_dir PATH` | Yes | Where to write outputs (created if absent) |
| `--top_k INT` | No | Print the top-k most similar pairs to stdout |
| `--additional_dir PATH` | No | Extra folder of PDFs to include (e.g. previous year) |

### Output files

| File | Description |
|---|---|
| `similarity_matrix.csv` | Pairwise cosine similarity with filenames as row/column headers |
| `similarity_matrix.pdf` | Colour-coded heatmap of the similarity matrix |
| `embeddings_cache.npz` | Cached document embeddings for faster re-runs |

### Examples

Basic run with top-5 flagging:

```bash
simcheck \
  --input_dir ./reports \
  --model_path ./models/embedding.gguf \
  --output_dir ./output \
  --top_k 5
```

Compare current and previous year reports:

```bash
simcheck \
  --input_dir ./reports_2026 \
  --additional_dir ./reports_2025 \
  --model_path ./models/embedding.gguf \
  --output_dir ./output \
  --top_k 10
```

> [!TIP]
> When using `--additional_dir`, if that folder already contains an `embeddings_cache.npz` from a prior run, those cached vectors are reused automatically (and never modified).

## How it works

```
pdf_utils.py  →  embedding.py  →  similarity.py
   (text)          (vectors)        (matrix + output)
```

1. **Extract** — PDF text is extracted with PyMuPDF and split into ~400-word chunks
2. **Embed** — Each chunk is embedded via `llama-cpp-python`, then mean-pooled into a single L2-normalised document vector
3. **Compare** — A pairwise cosine similarity matrix is computed (`mat @ mat.T` since vectors are pre-normalised)
4. **Output** — The matrix is written as CSV and rendered as a heatmap PDF

## Development

### Project structure

```
main.py          # CLI entry point; orchestrates the pipeline
pdf_utils.py     # PDF text extraction and chunking
embedding.py     # GGUF model loading, chunk embedding, caching
similarity.py    # Cosine similarity matrix, CSV/PDF output, top-k pairs
tests/           # Unit and end-to-end tests
sample_pdfs/     # Sample PDF files for testing
```

### Running tests

Both test files patch `llama_cpp` so no real GGUF model is needed:

```bash
# Unit tests
python tests/test_units.py

# End-to-end pipeline test
python tests/test_pipeline.py
```
