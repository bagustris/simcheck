"""
End-to-end smoke test for simcheck. llama_cpp is mocked via conftest.py.
"""
import csv
import os
import shutil
import time

import numpy as np
import pytest

import pdf_utils
import embedding
import similarity

SAMPLE = os.path.join(os.path.dirname(__file__), "sample_pdfs")
OUTPUT = "/tmp/simcheck_test_output"
FAKE_MODEL = "/tmp/fake.gguf"


@pytest.fixture(scope="module", autouse=True)
def setup_output():
    if os.path.exists(OUTPUT):
        shutil.rmtree(OUTPUT)
    os.makedirs(OUTPUT, exist_ok=True)
    open(FAKE_MODEL, "w").close()


@pytest.fixture(scope="module")
def pdf_paths():
    return sorted([
        os.path.join(SAMPLE, f)
        for f in os.listdir(SAMPLE)
        if f.endswith(".pdf")
    ])


@pytest.fixture(scope="module")
def model():
    return embedding.load_model(FAKE_MODEL)


@pytest.fixture(scope="module")
def vectors(model, pdf_paths):
    cache_path = os.path.join(OUTPUT, "embeddings_cache.npz")
    return embedding.load_or_compute(model, pdf_paths, cache_path)


@pytest.fixture(scope="module")
def similarity_data(vectors):
    return similarity.cosine_matrix(vectors)


def test_pdf_extraction():
    for fname in os.listdir(SAMPLE):
        path = os.path.join(SAMPLE, fname)
        text = pdf_utils.extract_text(path)
        assert len(text) > 10, f"extract_text({fname}) returned too little text"
        chunks = pdf_utils.chunk_text(text, chunk_words=50)
        assert len(chunks) >= 1, f"chunk_text({fname}) returned no chunks"


def test_load_or_compute_one_vector_per_pdf(vectors, pdf_paths):
    assert len(vectors) == len(pdf_paths)


def test_vectors_are_1d_float32_normalised(vectors):
    for fname, vec in vectors.items():
        assert vec.ndim == 1, f"{fname}: not 1-D"
        assert vec.dtype == np.float32, f"{fname}: not float32"
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5, f"{fname}: not unit-normalised"


def test_cache_written(pdf_paths):
    cache_path = os.path.join(OUTPUT, "embeddings_cache.npz")
    assert os.path.exists(cache_path)


def test_cache_not_rewritten_on_second_run(model, pdf_paths):
    cache_path = os.path.join(OUTPUT, "embeddings_cache.npz")
    mtime_before = os.path.getmtime(cache_path)
    time.sleep(0.05)
    embedding.load_or_compute(model, pdf_paths, cache_path)
    assert os.path.getmtime(cache_path) == mtime_before


def test_cached_vectors_match(model, pdf_paths, vectors):
    cache_path = os.path.join(OUTPUT, "embeddings_cache.npz")
    vectors2 = embedding.load_or_compute(model, pdf_paths, cache_path)
    for k in vectors:
        assert np.allclose(vectors[k], vectors2[k]), f"Cached vector for {k} changed"


def test_matrix_shape(similarity_data):
    labels, matrix = similarity_data
    n = len(labels)
    assert matrix.shape == (n, n)
    assert matrix.dtype == np.float32


def test_matrix_diagonal(similarity_data):
    labels, matrix = similarity_data
    for i in range(len(labels)):
        assert abs(matrix[i, i] - 1.0) < 1e-5, f"Diagonal [{i},{i}] != 1.0"


def test_matrix_range_and_symmetry(similarity_data):
    _, matrix = similarity_data
    assert np.all(matrix >= -1.0) and np.all(matrix <= 1.0)
    assert np.allclose(matrix, matrix.T)


def test_csv_output(similarity_data):
    labels, matrix = similarity_data
    csv_path = os.path.join(OUTPUT, "similarity_matrix.csv")
    similarity.save_csv(labels, matrix, csv_path)
    assert os.path.exists(csv_path)
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == ""
    assert rows[0][1:] == labels
    for i, label in enumerate(labels):
        assert rows[i + 1][0] == label
        assert abs(float(rows[i + 1][i + 1]) - 1.0) < 1e-4


def test_pdf_heatmap_output(similarity_data):
    labels, matrix = similarity_data
    pdf_path = os.path.join(OUTPUT, "similarity_matrix.pdf")
    similarity.save_pdf(labels, matrix, pdf_path)
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 1000


def test_top_k_pairs(similarity_data):
    labels, matrix = similarity_data
    pairs = similarity.top_k_pairs(labels, matrix, k=2)
    assert len(pairs) == 2
    assert pairs[0][2] >= pairs[1][2]
    for doc_a, doc_b, score in pairs:
        assert doc_a != doc_b
        assert 0.0 <= score <= 1.0


def test_stability_across_runs(model, pdf_paths, similarity_data):
    labels, matrix = similarity_data
    cache_path = os.path.join(OUTPUT, "embeddings_cache.npz")
    vectors3 = embedding.load_or_compute(model, pdf_paths, cache_path)
    labels3, matrix3 = similarity.cosine_matrix(vectors3)
    assert labels3 == labels
    assert np.allclose(matrix, matrix3)
