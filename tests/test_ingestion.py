import pytest
from pydantic import ValidationError

from mcqgen.ingestion import chunk_pages, clean_pages
from mcqgen.schemas import Draft


def test_cleaner_preserves_body_and_removes_repeated_boundaries():
    pages = [f"History\nChapter\nUnique content {number}\nFooter\n{number}" for number in range(3)]
    assert clean_pages(pages) == [f"Unique content {number}" for number in range(3)]


def test_chunks_keep_page_local_evidence():
    pages = ["First page facts. " * 30, "Second page facts. " * 30]
    chunks = chunk_pages(pages, "book.pdf", size=120, overlap=20)
    assert {chunk.page for chunk in chunks} == {1, 2}
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert all(chunk.text in pages[chunk.page - 1] for chunk in chunks)
    assert all(len(chunk.text) <= 120 for chunk in chunks)


def test_duplicate_options_rejected():
    with pytest.raises(ValidationError):
        Draft(
            question="Which answer is correct?",
            options=["A", "a", "B", "C"],
            answer_index=0,
            explanation="The source states this.",
            difficulty="easy",
            quote="The source states this.",
        )
