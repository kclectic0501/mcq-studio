from typing import TypeVar

import pytest
from pydantic import BaseModel

from mcqgen.config import Settings
from mcqgen.graph import build_graph, evidence_issues
from mcqgen.llm import ModelCallError, ModelConfigError
from mcqgen.schemas import Chunk, Critique, Draft, Request
from mcqgen.service import generate_quiz

Response = TypeVar("Response", bound=BaseModel)
CHUNK = Chunk(
    id="c1",
    document="book.pdf",
    page=2,
    text="The Earth orbits the Sun. The Moon orbits the Earth.",
)


class FakeModel:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.generations = 0
        self.judgments = 0

    def complete(self, prompt: str, schema: type[Response]) -> Response:
        if schema is Critique:
            self.judgments += 1
            return schema.model_validate({"passed": not self.fail, "issues": []})
        self.generations += 1
        return schema.model_validate(
            {
                "question": "Which star does the Earth orbit?",
                "options": ["Sun", "Sirius", "Vega", "Polaris"],
                "answer_index": 0,
                "explanation": "The text states that the Earth orbits the Sun.",
                "difficulty": "medium",
                "quote": "The Earth orbits the Sun.",
            }
        )


def test_accepts_and_preserves_answer_when_shuffling():
    model = FakeModel()
    quiz = build_graph(model, model).invoke(
        {
            "chunks": [CHUNK],
            "request": Request(count=1),
            "results": [],
        }
    )["quiz"]
    assert len(quiz.questions) == 1
    question = quiz.questions[0]
    assert question.options[question.answer_index] == "Sun"
    assert question.source.page == 2
    assert model.generations == model.judgments == 1


def test_bounded_retries_and_no_invalid_questions():
    model = FakeModel(fail=True)
    quiz = build_graph(model, model, max_retries=2).invoke(
        {
            "chunks": [CHUNK],
            "request": Request(count=1),
            "results": [],
        }
    )["quiz"]
    assert not quiz.questions
    assert quiz.warnings
    assert model.generations == model.judgments == 3


def test_parallel_fanout_and_dedup():
    model = FakeModel()
    quiz = build_graph(model, model).invoke(
        {
            "chunks": [CHUNK],
            "request": Request(count=3),
            "results": [],
        },
        config={"max_concurrency": 1},
    )["quiz"]
    assert model.generations == 3
    assert len(quiz.questions) == 1
    assert quiz.warnings


def test_quote_must_exist_in_source():
    draft = FakeModel().complete("", Draft)
    draft.quote = "An unsupported statement."
    assert evidence_issues(draft, CHUNK, Request())


def test_service_streams_progress_and_final_quiz(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("mcqgen.service.load_documents", lambda paths: [CHUNK])
    model = FakeModel()
    updates = list(
        generate_quiz(["book.pdf"], Request(count=1), Settings(_env_file=None), model, model)
    )
    assert updates[0] == (0, None)
    assert updates[1] == (1, None)
    assert updates[-1][1] is not None
    assert len(updates[-1][1].questions) == 1


@pytest.mark.parametrize("stage", ["generator", "judge"])
def test_provider_failure_preserves_other_jobs(stage):
    class FailingModel(FakeModel):
        def complete(self, prompt, schema):
            if (stage == "generator" and "Variation seed: 0." in prompt) or (
                stage == "judge" and self.judgments == 0
            ):
                self.judgments += 1
                raise ModelCallError("Provider request failed (HTTP 429)")
            return super().complete(prompt, schema)

    generator = FailingModel() if stage == "generator" else FakeModel()
    judge = FailingModel() if stage == "judge" else FakeModel()
    quiz = build_graph(generator, judge).invoke(
        {
            "chunks": [CHUNK],
            "request": Request(count=2),
            "results": [],
        },
        config={"max_concurrency": 1},
    )["quiz"]
    assert len(quiz.questions) == 1
    assert any("429" in warning for warning in quiz.warnings)


def test_configuration_failure_aborts_run():
    class BadConfigModel(FakeModel):
        def complete(self, prompt, schema):
            raise ModelConfigError("Invalid credentials")

    with pytest.raises(ModelConfigError, match="Invalid credentials"):
        build_graph(BadConfigModel(), FakeModel()).invoke(
            {
                "chunks": [CHUNK],
                "request": Request(count=1),
                "results": [],
            }
        )
