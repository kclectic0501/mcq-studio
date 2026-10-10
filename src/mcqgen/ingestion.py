import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from mcqgen.schemas import Chunk


def normalize(text: str) -> str:
    return " ".join(text.split())


def clean_pages(pages: list[str]) -> list[str]:
    boundaries: Counter[str] = Counter()
    for page in pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        boundaries.update(set(lines[:2] + lines[-2:]))
    repeated = {
        line
        for line, count in boundaries.items()
        if len(pages) >= 3 and count >= max(3, (len(pages) + 1) // 2)
    }
    result = []
    for page in pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        kept = [
            line
            for index, line in enumerate(lines)
            if not (
                (index < 2 or index >= len(lines) - 2)
                and (line in repeated or re.fullmatch(r"\d+", line))
            )
        ]
        result.append(normalize("\n".join(kept)))
    return result


def chunk_pages(
    pages: list[str], document: str, size: int = 3500, overlap: int = 250
) -> list[Chunk]:
    if size < 100 or not 0 <= overlap < size:
        raise ValueError("Chunk size must be >=100 and overlap smaller than size")
    chunks = []
    for page_number, text in enumerate(pages, start=1):
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start + size // 2, end)
                if boundary > start:
                    end = boundary
            content = text[start:end].strip()
            if content:
                chunks.append(
                    Chunk(
                        id=f"{document}:{page_number}:{start}",
                        document=document,
                        page=page_number,
                        text=content,
                    )
                )
            if end == len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks


def load_documents(paths: list[str], max_pages: int = 100) -> list[Chunk]:
    if not paths:
        raise ValueError("Upload at least one PDF")
    chunks = []
    total = 0
    for document_index, path in enumerate(paths):
        file = Path(path)
        if file.suffix.lower() != ".pdf":
            raise ValueError("Only PDF documents are supported")
        if file.stat().st_size > 25 * 1024 * 1024:
            raise ValueError(f"{file.name} exceeds the 25 MB file limit")
        reader = PdfReader(file)
        if reader.is_encrypted:
            raise ValueError(f"{file.name} is encrypted; upload an unlocked PDF")
        total += len(reader.pages)
        if total > max_pages:
            raise ValueError(f"Uploads must contain at most {max_pages} pages in total")
        pages = clean_pages([page.extract_text() or "" for page in reader.pages])
        if not any(page.strip() for page in pages):
            raise ValueError(f"{file.name} has no extractable text; OCR is not supported yet")
        document_chunks = chunk_pages(pages, file.name)
        for chunk in document_chunks:
            chunk.id = f"{document_index}:{chunk.id}"
        chunks.extend(document_chunks)
    return chunks
