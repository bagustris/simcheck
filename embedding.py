import os
import numpy as np
from tqdm import tqdm
from llama_cpp import Llama

from pdf_utils import extract_text, chunk_text


def load_model(model_path: str) -> Llama:
    """Load and return a llama.cpp model in embedding mode."""
    return Llama(
        model_path=model_path,
        embedding=True,
        n_ctx=512,
        n_gpu_layers=0,
        verbose=False,
    )


def embed_chunks(model: Llama, chunks: list[str]) -> np.ndarray:
    """Return (N, dim) float32 array of L2-normalised embeddings for each chunk."""
    embeddings = []
    for chunk in chunks:
        result = model.embed(chunk)
        vec = np.array(result, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        embeddings.append(vec)
    return np.stack(embeddings, axis=0)


def document_vector(model: Llama, chunks: list[str]) -> np.ndarray:
    """Return a single (dim,) float32 vector via mean pooling over chunk embeddings."""
    chunk_vecs = embed_chunks(model, chunks)
    mean_vec = chunk_vecs.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm
    return mean_vec.astype(np.float32)


def load_or_compute(
    model: Llama,
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
    # Load read-only cache
    readonly_cache: dict[str, np.ndarray] = {}
    if readonly_cache_path and os.path.exists(readonly_cache_path):
        data = np.load(readonly_cache_path, allow_pickle=False)
        for key in data.files:
            readonly_cache[key] = data[key]

    # Load read-write cache
    rw_cache: dict[str, np.ndarray] = {}
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        for key in data.files:
            rw_cache[key] = data[key]

    result: dict[str, np.ndarray] = {}
    to_compute: list[str] = []

    for pdf_path in pdf_paths:
        fname = os.path.basename(pdf_path)
        if fname in readonly_cache:
            result[fname] = readonly_cache[fname]
        elif fname in rw_cache:
            result[fname] = rw_cache[fname]
        else:
            to_compute.append(pdf_path)

    if to_compute:
        for pdf_path in tqdm(to_compute, desc="Embedding", unit="doc"):
            fname = os.path.basename(pdf_path)
            text = extract_text(pdf_path)
            chunks = chunk_text(text)
            if not chunks:
                # Empty document: zero vector placeholder (dim unknown until first real embed)
                chunks = [""]
            vec = document_vector(model, chunks)
            result[fname] = vec
            rw_cache[fname] = vec

        # Save updated rw cache
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path, **rw_cache)

    return result
