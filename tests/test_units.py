"""
Unit tests for simcheck modules. llama_cpp is mocked via conftest.py.
"""
import os
import tempfile
import argparse

import numpy as np
import pytest

import pdf_utils
import embedding
import similarity
import main as main_module

from conftest import DIM

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_pdfs")


# ── pdf_utils.chunk_text ──────────────────────────────────────────────────────

def test_chunk_text_empty():
    assert pdf_utils.chunk_text("") == []

def test_chunk_text_single_word():
    assert pdf_utils.chunk_text("hello") == ["hello"]

def test_chunk_text_exact_boundary():
    words = " ".join(f"w{i}" for i in range(8))
    chunks = pdf_utils.chunk_text(words, chunk_words=4)
    assert len(chunks) == 2
    assert all(len(c.split()) == 4 for c in chunks)

def test_chunk_text_non_exact_boundary():
    words = " ".join(f"w{i}" for i in range(9))
    chunks = pdf_utils.chunk_text(words, chunk_words=4)
    assert len(chunks) == 3
    assert len(chunks[-1].split()) == 1


# ── embedding.embed_chunks ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    return embedding.load_model("/tmp/unused.gguf")

def test_embed_chunks_shape(model):
    vecs = embedding.embed_chunks(model, ["hello world", "foo bar baz"])
    assert vecs.shape == (2, DIM)

def test_embed_chunks_dtype(model):
    vecs = embedding.embed_chunks(model, ["hello world", "foo bar baz"])
    assert vecs.dtype == np.float32

def test_embed_chunks_unit_normalised(model):
    vecs = embedding.embed_chunks(model, ["hello world", "foo bar baz"])
    for i in range(2):
        assert abs(float(np.linalg.norm(vecs[i])) - 1.0) < 1e-5


# ── embedding.document_vector ─────────────────────────────────────────────────

def test_document_vector_1d(model):
    v = embedding.document_vector(model, ["chunk one", "chunk two", "chunk three"])
    assert v.ndim == 1

def test_document_vector_shape(model):
    v = embedding.document_vector(model, ["chunk one", "chunk two", "chunk three"])
    assert v.shape == (DIM,)

def test_document_vector_dtype(model):
    v = embedding.document_vector(model, ["chunk one", "chunk two", "chunk three"])
    assert v.dtype == np.float32

def test_document_vector_normalised(model):
    v = embedding.document_vector(model, ["chunk one", "chunk two", "chunk three"])
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5

def test_document_vector_single_chunk(model):
    v = embedding.document_vector(model, ["only chunk"])
    assert v.ndim == 1 and v.shape == (DIM,)


# ── embedding.load_or_compute with readonly_cache_path ───────────────────────

def test_load_or_compute_readonly_cache(model):
    pdf_paths = sorted([
        os.path.join(SAMPLE_DIR, f)
        for f in os.listdir(SAMPLE_DIR)
        if f.endswith(".pdf")
    ])
    with tempfile.TemporaryDirectory() as tmpdir:
        readonly_dir = os.path.join(tmpdir, "readonly")
        rw_dir = os.path.join(tmpdir, "rw")
        os.makedirs(readonly_dir)
        os.makedirs(rw_dir)

        readonly_cache = os.path.join(readonly_dir, "embeddings_cache.npz")
        rw_cache = os.path.join(rw_dir, "embeddings_cache.npz")

        first_pdf = pdf_paths[0]
        first_fname = os.path.basename(first_pdf)
        text = pdf_utils.extract_text(first_pdf)
        chunks = pdf_utils.chunk_text(text)
        vec = embedding.document_vector(model, chunks or [""])
        np.savez(readonly_cache, **{first_fname: vec})

        vectors = embedding.load_or_compute(model, pdf_paths, rw_cache, readonly_cache_path=readonly_cache)
        assert len(vectors) == len(pdf_paths)
        assert np.allclose(vectors[first_fname], vec)

def test_readonly_cache_not_written(model):
    pdf_paths = sorted([
        os.path.join(SAMPLE_DIR, f)
        for f in os.listdir(SAMPLE_DIR)
        if f.endswith(".pdf")
    ])
    with tempfile.TemporaryDirectory() as tmpdir:
        readonly_dir = os.path.join(tmpdir, "readonly")
        rw_dir = os.path.join(tmpdir, "rw")
        os.makedirs(readonly_dir)
        os.makedirs(rw_dir)

        readonly_cache = os.path.join(readonly_dir, "embeddings_cache.npz")
        rw_cache = os.path.join(rw_dir, "embeddings_cache.npz")

        first_fname = os.path.basename(pdf_paths[0])
        text = pdf_utils.extract_text(pdf_paths[0])
        chunks = pdf_utils.chunk_text(text)
        vec = embedding.document_vector(model, chunks or [""])
        np.savez(readonly_cache, **{first_fname: vec})

        embedding.load_or_compute(model, pdf_paths, rw_cache, readonly_cache_path=readonly_cache)
        mtime_before = os.path.getmtime(readonly_cache)
        import time; time.sleep(0.05)
        embedding.load_or_compute(model, pdf_paths, rw_cache, readonly_cache_path=readonly_cache)
        assert os.path.getmtime(readonly_cache) == mtime_before


# ── similarity.top_k_pairs edge cases ────────────────────────────────────────

@pytest.fixture
def sample_matrix():
    labels = ["a", "b", "c"]
    mat = np.array([
        [1.0, 0.9, 0.5],
        [0.9, 1.0, 0.3],
        [0.5, 0.3, 1.0],
    ], dtype=np.float32)
    return labels, mat

def test_top_k_zero(sample_matrix):
    labels, mat = sample_matrix
    assert similarity.top_k_pairs(labels, mat, k=0) == []

def test_top_k_exceeds_pairs(sample_matrix):
    labels, mat = sample_matrix
    pairs = similarity.top_k_pairs(labels, mat, k=100)
    assert len(pairs) == 3

def test_top_k_no_self_pairs(sample_matrix):
    labels, mat = sample_matrix
    pairs = similarity.top_k_pairs(labels, mat, k=100)
    assert all(a != b for a, b, _ in pairs)

def test_top_k_sorted(sample_matrix):
    labels, mat = sample_matrix
    pairs = similarity.top_k_pairs(labels, mat, k=100)
    assert pairs[0][2] >= pairs[1][2] >= pairs[2][2]

def test_top_k_single_document():
    labels = ["only"]
    mat = np.array([[1.0]], dtype=np.float32)
    assert similarity.top_k_pairs(labels, mat, k=5) == []


# ── main.collect_pdfs ─────────────────────────────────────────────────────────

def test_collect_pdfs_filters_non_pdf():
    with tempfile.TemporaryDirectory() as tmpdir:
        for name in ["b.pdf", "a.pdf", "c.PDF", "readme.txt"]:
            open(os.path.join(tmpdir, name), "w").close()
        collected = main_module.collect_pdfs(tmpdir)
        basenames = [os.path.basename(p) for p in collected]
        assert "readme.txt" not in basenames
        assert basenames == sorted(basenames)
        assert len(basenames) == 3

def test_collect_pdfs_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert main_module.collect_pdfs(tmpdir) == []


# ── main.validate_args ────────────────────────────────────────────────────────

def make_args(**kwargs):
    return argparse.Namespace(
        input_dir=kwargs.get("input_dir", "/nonexistent"),
        model_path=kwargs.get("model_path", "/nonexistent.gguf"),
        additional_dir=kwargs.get("additional_dir", None),
    )

def test_validate_args_valid():
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_model = os.path.join(tmpdir, "model.gguf")
        open(fake_model, "w").close()
        main_module.validate_args(make_args(input_dir=tmpdir, model_path=fake_model))

def test_validate_args_missing_input_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_model = os.path.join(tmpdir, "model.gguf")
        open(fake_model, "w").close()
        with pytest.raises(SystemExit):
            main_module.validate_args(make_args(input_dir="/no/such/dir", model_path=fake_model))

def test_validate_args_missing_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(SystemExit):
            main_module.validate_args(make_args(input_dir=tmpdir, model_path="/no/model.gguf"))

def test_validate_args_missing_additional_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_model = os.path.join(tmpdir, "model.gguf")
        open(fake_model, "w").close()
        with pytest.raises(SystemExit):
            main_module.validate_args(make_args(
                input_dir=tmpdir, model_path=fake_model,
                additional_dir="/no/such/additional",
            ))

def test_validate_args_valid_additional_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_model = os.path.join(tmpdir, "model.gguf")
        open(fake_model, "w").close()
        extra_dir = os.path.join(tmpdir, "extra")
        os.makedirs(extra_dir)
        main_module.validate_args(make_args(
            input_dir=tmpdir, model_path=fake_model, additional_dir=extra_dir,
        ))
