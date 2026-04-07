"""
End-to-end smoke test for simcheck.
Patches llama_cpp.Llama so no real GGUF model is needed.
"""
import os
import shutil
import sys
import numpy as np
import types

# ── Patch llama_cpp before any module imports it ─────────────────────────────
RNG = np.random.default_rng(42)
DIM = 64  # small fixed dimension


class FakeLlama:
    def __init__(self, *args, **kwargs):
        pass

    def embed(self, text: str):
        # Deterministic based on text content
        seed = sum(ord(c) for c in text) % (2**31)
        rng = np.random.default_rng(seed)
        vec = rng.random(DIM).astype(np.float32)
        return vec.tolist()


fake_llama_cpp = types.ModuleType("llama_cpp")
fake_llama_cpp.Llama = FakeLlama
sys.modules["llama_cpp"] = fake_llama_cpp

# ── Now import project modules ────────────────────────────────────────────────
import pdf_utils
import embedding
import similarity

# ── Helpers ──────────────────────────────────────────────────────────────────
OUTPUT = "/tmp/simcheck_test_output"
SAMPLE = "/home/bagus/github/simcheck/sample_pdfs"
FAKE_MODEL = "/tmp/fake.gguf"

if os.path.exists(OUTPUT):
    shutil.rmtree(OUTPUT)
os.makedirs(OUTPUT, exist_ok=True)
open(FAKE_MODEL, "w").close()  # empty placeholder


def check(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    print(f"PASS: {msg}")


# ── 1. PDF extraction ─────────────────────────────────────────────────────────
for fname in os.listdir(SAMPLE):
    path = os.path.join(SAMPLE, fname)
    text = pdf_utils.extract_text(path)
    check(len(text) > 10, f"extract_text({fname}) returns non-empty text")
    chunks = pdf_utils.chunk_text(text, chunk_words=50)
    check(len(chunks) >= 1, f"chunk_text({fname}) returns at least one chunk")

# ── 2. Embedding & caching ────────────────────────────────────────────────────
model = embedding.load_model(FAKE_MODEL)
pdf_paths = sorted([os.path.join(SAMPLE, f) for f in os.listdir(SAMPLE) if f.endswith(".pdf")])
cache_path = os.path.join(OUTPUT, "embeddings_cache.npz")

vectors = embedding.load_or_compute(model, pdf_paths, cache_path)

check(len(vectors) == len(pdf_paths), "load_or_compute returns one vector per PDF")
for fname, vec in vectors.items():
    check(vec.ndim == 1, f"vector for {fname} is 1-D")
    check(vec.dtype == np.float32, f"vector for {fname} is float32")
    norm = float(np.linalg.norm(vec))
    check(abs(norm - 1.0) < 1e-5, f"vector for {fname} is unit-normalised (norm={norm:.6f})")

check(os.path.exists(cache_path), "embeddings_cache.npz written")

# ── 3. Cache is used on second run (no new computation) ───────────────────────
mtime_before = os.path.getmtime(cache_path)
import time; time.sleep(0.05)
vectors2 = embedding.load_or_compute(model, pdf_paths, cache_path)
mtime_after = os.path.getmtime(cache_path)
check(mtime_before == mtime_after, "Cache not rewritten on second run (mtime unchanged)")
for k in vectors:
    check(np.allclose(vectors[k], vectors2[k]), f"Cached vector for {k} matches original")

# ── 4. Similarity matrix ──────────────────────────────────────────────────────
labels, matrix = similarity.cosine_matrix(vectors)
n = len(labels)
check(matrix.shape == (n, n), f"Matrix shape is ({n},{n})")
check(matrix.dtype == np.float32, "Matrix dtype is float32")
for i in range(n):
    check(abs(matrix[i, i] - 1.0) < 1e-5, f"Diagonal [{i},{i}] == 1.0")
check(np.all(matrix >= -1.0) and np.all(matrix <= 1.0), "All values in [-1, 1]")
check(np.allclose(matrix, matrix.T), "Matrix is symmetric")

# ── 5. CSV output ─────────────────────────────────────────────────────────────
csv_path = os.path.join(OUTPUT, "similarity_matrix.csv")
similarity.save_csv(labels, matrix, csv_path)
check(os.path.exists(csv_path), "similarity_matrix.csv written")
import csv
with open(csv_path) as f:
    rows = list(csv.reader(f))
check(rows[0][0] == "", "CSV header first cell is empty")
check(rows[0][1:] == labels, "CSV column headers match labels")
for i, label in enumerate(labels):
    check(rows[i + 1][0] == label, f"CSV row {i+1} label is '{label}'")
    val = float(rows[i + 1][i + 1])
    check(abs(val - 1.0) < 1e-4, f"CSV diagonal [{i},{i}] ≈ 1.0 (got {val})")

# ── 6. PDF heatmap output ─────────────────────────────────────────────────────
pdf_path = os.path.join(OUTPUT, "similarity_matrix.pdf")
similarity.save_pdf(labels, matrix, pdf_path)
check(os.path.exists(pdf_path), "similarity_matrix.pdf written")
check(os.path.getsize(pdf_path) > 1000, "similarity_matrix.pdf has non-trivial size")

# ── 7. Top-k pairs ────────────────────────────────────────────────────────────
pairs = similarity.top_k_pairs(labels, matrix, k=2)
check(len(pairs) == 2, "top_k_pairs returns k=2 pairs")
check(pairs[0][2] >= pairs[1][2], "top_k_pairs sorted descending")
for doc_a, doc_b, score in pairs:
    check(doc_a != doc_b, f"Pair ({doc_a}, {doc_b}) is not self-pair")
    check(0.0 <= score <= 1.0, f"Score {score:.4f} in [0,1]")

# ── 8. Stability across runs ──────────────────────────────────────────────────
vectors3 = embedding.load_or_compute(model, pdf_paths, cache_path)
labels3, matrix3 = similarity.cosine_matrix(vectors3)
check(labels3 == labels, "Labels stable across runs")
check(np.allclose(matrix, matrix3), "Matrix stable across runs")

print("\nAll tests passed.")
