"""
ingestion.py
------------
Handles PDF loading, text extraction, and chunking.
Strategy: Overlapping chunks of 512 tokens with 50-token overlap
to preserve context across chunk boundaries.
"""

import fitz  # PyMuPDF
import os
import re
from dataclasses import dataclass
from typing import List


@dataclass
class DocumentChunk:
    """Represents a single chunk of text from a document."""
    text: str
    source: str        # filename
    page: int          # page number (1-indexed)
    chunk_id: str      # unique identifier
    char_start: int    # character offset in original doc


def extract_text_from_pdf(pdf_path: str) -> List[dict]:
    """
    Extract text page-by-page from a PDF file.
    Returns list of {page_num, text} dicts.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()

        if text:  # skip blank pages
            pages.append({
                "page_num": page_num + 1,
                "text": text
            })

    doc.close()
    return pages


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[str]:
    """
    Split text into overlapping chunks by word count.
    This preserves context at chunk boundaries.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # slide with overlap

    return chunks


def ingest_pdf(pdf_path: str) -> List[DocumentChunk]:
    """
    Full ingestion pipeline: extract → clean → chunk.
    Returns a list of DocumentChunk objects ready for embedding.
    """
    filename = os.path.basename(pdf_path)
    pages = extract_text_from_pdf(pdf_path)

    all_chunks = []
    chunk_counter = 0

    for page_data in pages:
        page_num = page_data["page_num"]
        page_text = page_data["text"]

        raw_chunks = chunk_text(page_text, chunk_size=512, overlap=50)

        for i, chunk_text_str in enumerate(raw_chunks):
            if len(chunk_text_str.strip()) < 50:
                continue  # skip tiny meaningless chunks

            chunk = DocumentChunk(
                text=chunk_text_str,
                source=filename,
                page=page_num,
                chunk_id=f"{filename}_p{page_num}_c{chunk_counter}",
                char_start=i * 450  # approximate character offset
            )
            all_chunks.append(chunk)
            chunk_counter += 1

    print(f"✅ Ingested '{filename}': {len(pages)} pages → {len(all_chunks)} chunks")
    return all_chunks


def ingest_folder(folder_path: str) -> List[DocumentChunk]:
    """
    Ingest all PDFs from a folder.
    """
    all_chunks = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"⚠️  No PDF files found in {folder_path}")
        return []

    for pdf_file in pdf_files:
        full_path = os.path.join(folder_path, pdf_file)
        try:
            chunks = ingest_pdf(full_path)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"❌ Error processing {pdf_file}: {e}")

    return all_chunks


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        chunks = ingest_pdf(sys.argv[1])
        print(f"\nFirst chunk preview:\n{chunks[0].text[:300]}...")
