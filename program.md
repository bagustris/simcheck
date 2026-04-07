# simcheck — Student Report Similarity Tool

Build a Python command-line tool that detects semantic similarity between student PDF reports using local embeddings. The tool must run entirely offline, produce a pairwise similarity matrix, and flag the most similar document pairs.

## Success Criteria

- [ ] `python main.py --input_dir ./reports --model_path ./models/emb.gguf --output_dir ./output` runs end-to-end without errors
- [ ] `output/similarity_matrix.csv` is written with PDF filenames as row/column headers and cosine similarity values (0.0–1.0); diagonal is exactly 1.0
- [ ] `output/similarity_matrix.pdf` is written with a colour-coded heatmap rendering of the same matrix
- [ ] When `--top_k N` is given, the N most similar non-self pairs are printed to stdout, sorted descending by score
- [ ] When `--additional_dir` is given, PDFs from that folder are included in the matrix alongside `--input_dir` PDFs
- [ ] Embeddings are cached to `output/embeddings_cache.npz`; re-running skips files whose cache entry already exists
- [ ] When `--additional_dir` is given and that folder contains `embeddings_cache.npz`, those cached vectors are loaded directly without re-embedding (the file is never modified)
- [ ] The tool handles 10–100 PDFs on CPU without running out of memory
- [ ] All similarity scores are stable across runs (same inputs → same outputs)

## Constraints

- **No external APIs.** Every computation runs locally.
- **Embeddings via llama.cpp only** (`llama-cpp-python` package). Model is a GGUF file provided by the user.
- **No GPU required.** Must work on CPU-only machines.
- **No unnecessary abstractions.** Prefer flat, readable code over frameworks and layers.
- **Do not modify the public CLI interface** described below.

## CLI Interface

```
python main.py \
  --input_dir PATH       # required: folder of student PDF files
  --model_path PATH      # required: path to a GGUF embedding model
  --output_dir PATH      # required: where to write outputs (created if absent)
  --top_k INT            # optional: print top-k most similar pairs (default: off)
  --additional_dir PATH  # optional: extra folder of PDFs (e.g., previous year)
```

Implemented with `argparse`. Validated on startup — exit with a clear error message if required paths do not exist.

## File Structure

```
main.py          # CLI entry point; orchestrates the pipeline
pdf_utils.py     # PDF text extraction and chunking
embedding.py     # llama.cpp embedding calls and mean-pooling
similarity.py    # cosine similarity matrix and top-k output
requirements.txt # pinned dependencies
```

## Module Contracts

### `pdf_utils.py`

```python
def extract_text(pdf_path: str) -> str:
    """Return clean, whitespace-normalised text from a PDF. Skip blank pages."""

def chunk_text(text: str, chunk_words: int = 400) -> list[str]:
    """Split text into chunks of ~chunk_words words. Return list of strings."""
```

### `embedding.py`

```python
def load_model(model_path: str) -> object:
    """Load and return a llama.cpp model in embedding mode."""

def embed_chunks(model, chunks: list[str]) -> np.ndarray:
    """Return (N, dim) float32 array of L2-normalised embeddings for each chunk."""

def document_vector(model, chunks: list[str]) -> np.ndarray:
    """Return a single (dim,) float32 vector via mean pooling over chunk embeddings."""

def load_or_compute(
    model,
    pdf_paths: list[str],
    cache_path: str,
    readonly_cache_path: str | None = None,
) -> dict[str, np.ndarray]:
    """Return {filename: vector} dict.

    Resolution order per file:
      1. readonly_cache_path (e.g. additional_dir/embeddings_cache.npz) — never written to.
      2. cache_path (output/embeddings_cache.npz) — read and updated.
      3. Compute from scratch and save to cache_path.
    """
```

### `similarity.py`

```python
def cosine_matrix(vectors: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Return (labels, NxN float32 matrix) of pairwise cosine similarities."""

def save_csv(labels: list[str], matrix: np.ndarray, path: str) -> None:
    """Write the matrix as CSV with labels as row/column headers."""

def save_pdf(labels: list[str], matrix: np.ndarray, path: str) -> None:
    """Write a colour-coded heatmap PDF of the similarity matrix using matplotlib."""

def top_k_pairs(labels: list[str], matrix: np.ndarray, k: int) -> list[tuple[str, str, float]]:
    """Return the k highest off-diagonal (doc_a, doc_b, score) tuples, sorted descending."""
```

## Pipeline (in `main.py`)

1. Parse and validate args.
2. Collect PDF paths from `--input_dir` (and `--additional_dir` if given).
3. Load the GGUF model via `embedding.load_model`.
4. Compute/load document vectors via `embedding.load_or_compute`, passing:
   - `cache_path` = `output_dir/embeddings_cache.npz` (read-write)
   - `readonly_cache_path` = `additional_dir/embeddings_cache.npz` if it exists (read-only; never overwritten)
5. Build cosine matrix via `similarity.cosine_matrix`.
6. Write CSV and PDF outputs.
7. If `--top_k` given, print top-k pairs to stdout.

Show a `tqdm` progress bar during embedding computation.

## Performance Notes

- Batch chunk embedding calls where llama.cpp allows.
- Use `numpy` for all matrix operations; avoid Python loops over the full matrix.
- Use `np.savez` / `np.load` for the embedding cache.
- Free model from memory after embedding is complete if possible.

## Dependencies (`requirements.txt`)

```
pymupdf
llama-cpp-python
numpy
matplotlib
tqdm
```

## Example Usage

```bash
# Basic run
python main.py \
  --input_dir ./reports \
  --model_path ./models/embedding.gguf \
  --output_dir ./output \
  --top_k 5

# Include previous year reports
python main.py \
  --input_dir ./reports_2026 \
  --additional_dir ./reports_2025 \
  --model_path ./models/embedding.gguf \
  --output_dir ./output \
  --top_k 10
```

## Verification

```bash
# Smoke test — must exit 0 and produce all three output files
python main.py --input_dir ./sample_pdfs --model_path ./models/embedding.gguf --output_dir ./out
ls ./out/similarity_matrix.csv ./out/similarity_matrix.pdf ./out/embeddings_cache.npz

# Second run must use cache (no embedding progress bar output)
python main.py --input_dir ./sample_pdfs --model_path ./models/embedding.gguf --output_dir ./out
```

## Out of Scope

- Web UI or REST API
- Support for non-PDF file formats
- Multi-language or multilingual models (assumed English)
- Automatic model download (user provides the GGUF file)
