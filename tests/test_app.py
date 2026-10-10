import gradio as gr
import pytest

from mcqgen.app import check_answer, create_app
from mcqgen.config import Settings
from mcqgen.llm import ModelConfigError
from mcqgen.schemas import MCQ, Chunk

def test_ui_constructs_without_credentials():
    app = create_app(Settings(_env_file=None))
    assert app.title == "MCQ Studio"

def test_answer_feedback_escapes_document_content():
    question = MCQ(
        question="Which star does the Earth orbit?",
        options=["Sun", "Vega", "Sirius", "Polaris"],
        answer_index=0,
        explanation="The text names the Sun.",
        difficulty="easy",
        quote="<script>alert('oops')</script>",
        source=chunk(id="1", document="book.pdf", page=1, text="The Earth orbits the Sun.")
    )
    assert "Select an answer" in check_answer(None, question)
    feedback = check_answer(0, question)
    assert "Correct" in feedback
    assert "<script>" not in feedback
    assert "&lt;script&gt;" in feedback

def test_ui_shows_configuration_error(monkeypatch):
    def failed_generation(*args):
        raise ModelConfigError("Invalid provider credentials")
    
    monkeypatch.setattr("mcqgen.app.generate_quiz", failed_generation)
    app = create_app(Settings(_env_file=None))
    callback = next(item.fn for item in app.fns.values() if item.fn.__name__ == "run")
    updates = callback(["sample.pdf"], 1, "medium")
    next(updates)
    with pytest.raises(gr.Error, match="Invalid provider credentials"):
        next(updates)

def test_cli_configuration_failure_has_clean_exit(monkeypatch, capsys):
    from mcqgen.app import main

    def failed_generation(*args):
        raise ModelConfigError("Invalid provider credentials")

        monkeypatch.setattr("sys.argv", ["mcqgen", "generate", "sample.pdf"])
        monkeypatch.setattr("mcqgen.app.generate_quiz", failed_generation)
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 1
        assert "Invalid provider credentials" in capsys.readouterr().err
