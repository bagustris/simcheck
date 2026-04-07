import re
import fitz  # pymupdf


def extract_text(pdf_path: str) -> str:
    """Return clean, whitespace-normalised text from a PDF. Skip blank pages."""
    doc = fitz.open(pdf_path)
    parts = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            parts.append(text)
    doc.close()
    combined = " ".join(parts)
    # Normalise whitespace
    combined = re.sub(r"\s+", " ", combined).strip()
    return combined


def chunk_text(text: str, chunk_words: int = 400) -> list[str]:
    """Split text into chunks of ~chunk_words words. Return list of strings."""
    words = text.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), chunk_words):
        chunk = " ".join(words[i : i + chunk_words])
        chunks.append(chunk)
    return chunks
