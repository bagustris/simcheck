import argparse
import os
import sys

import embedding
import similarity


def collect_pdfs(directory: str) -> list[str]:
    paths = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".pdf")
    ]
    return sorted(paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="simcheck — Student Report Similarity Tool"
    )
    parser.add_argument("--input_dir", required=True, help="Folder of student PDF files")
    parser.add_argument(
        "--model_path", required=True, help="Path to a GGUF embedding model"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Where to write outputs (created if absent)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Print top-k most similar pairs to stdout",
    )
    parser.add_argument(
        "--additional_dir",
        default=None,
        help="Extra folder of PDFs (e.g., previous year)",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.input_dir):
        sys.exit(f"Error: --input_dir '{args.input_dir}' does not exist or is not a directory.")
    if not os.path.isfile(args.model_path):
        sys.exit(f"Error: --model_path '{args.model_path}' does not exist or is not a file.")
    if args.additional_dir is not None and not os.path.isdir(args.additional_dir):
        sys.exit(
            f"Error: --additional_dir '{args.additional_dir}' does not exist or is not a directory."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Collect PDFs
    input_pdfs = collect_pdfs(args.input_dir)
    if not input_pdfs:
        sys.exit(f"Error: no PDF files found in '{args.input_dir}'.")

    additional_pdfs: list[str] = []
    if args.additional_dir:
        additional_pdfs = collect_pdfs(args.additional_dir)

    all_pdfs = input_pdfs + additional_pdfs

    # 2. Load model
    print(f"Loading model: {args.model_path}")
    model = embedding.load_model(args.model_path)

    # 3. Compute/load embeddings
    cache_path = os.path.join(args.output_dir, "embeddings_cache.npz")
    readonly_cache_path: str | None = None
    if args.additional_dir:
        candidate = os.path.join(args.additional_dir, "embeddings_cache.npz")
        if os.path.exists(candidate):
            readonly_cache_path = candidate

    vectors = embedding.load_or_compute(
        model, all_pdfs, cache_path, readonly_cache_path=readonly_cache_path
    )

    # Free model memory
    del model

    # 4. Build similarity matrix
    labels, matrix = similarity.cosine_matrix(vectors)

    # 5. Write outputs
    csv_path = os.path.join(args.output_dir, "similarity_matrix.csv")
    pdf_path = os.path.join(args.output_dir, "similarity_matrix.pdf")
    similarity.save_csv(labels, matrix, csv_path)
    print(f"Wrote {csv_path}")
    similarity.save_pdf(labels, matrix, pdf_path)
    print(f"Wrote {pdf_path}")
    print(f"Cache: {cache_path}")

    # 6. Top-k pairs
    if args.top_k is not None:
        pairs = similarity.top_k_pairs(labels, matrix, args.top_k)
        print(f"\nTop {args.top_k} most similar pairs:")
        for doc_a, doc_b, score in pairs:
            print(f"  {score:.4f}  {doc_a}  <->  {doc_b}")


if __name__ == "__main__":
    main()
