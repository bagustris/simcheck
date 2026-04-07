import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cosine_matrix(vectors: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Return (labels, NxN float32 matrix) of pairwise cosine similarities."""
    labels = sorted(vectors.keys())
    mat = np.stack([vectors[l] for l in labels], axis=0)  # (N, dim)
    # Vectors are already L2-normalised; dot product = cosine similarity
    sim = mat @ mat.T
    # Clip to [-1, 1] to handle floating point drift
    sim = np.clip(sim, -1.0, 1.0).astype(np.float32)
    return labels, sim


def save_csv(labels: list[str], matrix: np.ndarray, path: str) -> None:
    """Write the matrix as CSV with labels as row/column headers."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        for i, label in enumerate(labels):
            writer.writerow([label] + [f"{v:.6f}" for v in matrix[i]])


def save_pdf(labels: list[str], matrix: np.ndarray, path: str) -> None:
    """Write a colour-coded heatmap PDF of the similarity matrix using matplotlib."""
    n = len(labels)
    fig_size = max(6, n * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=max(6, 10 - n // 10))
    ax.set_yticklabels(labels, fontsize=max(6, 10 - n // 10))
    ax.set_title("Document Similarity Matrix")
    plt.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def top_k_pairs(
    labels: list[str], matrix: np.ndarray, k: int
) -> list[tuple[str, str, float]]:
    """Return the k highest off-diagonal (doc_a, doc_b, score) tuples, sorted descending."""
    n = len(labels)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((labels[i], labels[j], float(matrix[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:k]
