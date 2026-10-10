import operator
import random
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import ValidationError
from rapidfuzz.fuzz import ratio

from mcqgen.ingestion import normalize
from mcqgen.llm import Model, ModelCallError
from mcqgen.schemas import MCQ, Chunk, Critique, Draft, Quiz, Request


class Job(TypedDict):
    slot: int
    chunk: Chunk
    request: Request
    attempts: int
    feedback: str
    draft: Draft | None
    passed: bool


class Result(TypedDict):
    slot: int
    question: MCQ | None
    warning: str


class State(TypedDict):
    chunks: list[Chunk]
    request: Request
    results: Annotated[list[Result], operator.add]
    quiz: Quiz


def evidence_issues(draft: Draft, chunk: Chunk, request: Request) -> list[str]:
    issues = []
    if normalize(draft.quote) not in normalize(chunk.text):
        issues.append("Supporting quote must be copied verbatim from the source")
    if draft.difficulty != request.difficulty:
        issues.append("Difficulty must match the requested level")
    return issues


def build_graph(generator: Model, judge: Model, max_retries: int = 2) -> Any:
    if not 0 <= max_retries <= 5:
        raise ValueError("max_retries must be between 0 and 5")

    def generate(job: Job) -> dict[str, Any]:
        prompt = (
            f"Generate one single-correct MCQ, difficulty={job['request'].difficulty.value}. "
            "Use exactly four distinct plausible options. Include an explanation and a verbatim "
            "supporting quote. Easy: factual recall. Medium: explain a relationship. "
            "Hard: inference requiring reasoning from the source, not obscure trivia. "
            f"Variation seed: {job['slot']}. Choose a distinct fact or angle for this seed.\n"
            f"Previous issues: {job['feedback']}\n"
            f"<source>\n{job['chunk'].text}\n</source>"
        )
        try:
            draft = generator.complete(prompt, Draft)
            return {"draft": draft, "attempts": job["attempts"] + 1, "passed": False}
        except ModelCallError as error:
            return {
                "draft": None,
                "attempts": max_retries + 1,
                "passed": False,
                "feedback": str(error),
            }
        except (ValidationError, ValueError) as error:
            return {
                "draft": None,
                "attempts": job["attempts"] + 1,
                "passed": False,
                "feedback": f"Invalid output: {error}",
            }

    def validate(job: Job) -> dict[str, Any]:
        draft = job["draft"]
        if draft is None:
            return {"passed": False}
        issues = evidence_issues(draft, job["chunk"], job["request"])
        if not issues:
            try:
                critique = judge.complete(
                    "Audit this MCQ against the source only. Pass only if exactly one option "
                    "is correct, the indexed answer and explanation are supported, no other "
                    "option is defensibly correct, distractors are plausible, and difficulty "
                    "matches (easy=recall, medium=relationships, hard=reasoned inference). "
                    "Reject ambiguity and claims needing outside knowledge.\n"
                    f"<source>\n{job['chunk'].text}\n</source>\n"
                    f"MCQ: {draft.model_dump_json()}",
                    Critique,
                )
                if not critique.passed or critique.issues:
                    issues = critique.issues or ["Judge rejected the question"]
            except ModelCallError as error:
                return {"passed": False, "attempts": max_retries + 1, "feedback": str(error)}
            except (ValidationError, ValueError):
                issues = ["Judge returned invalid structured output"]
        return {"passed": not issues, "feedback": "; ".join(issues)}

    def retry(job: Job) -> str:
        return "generate" if not job["passed"] and job["attempts"] <= max_retries else END

    worker = StateGraph(Job)
    worker.add_node("generate", RunnableLambda(generate))
    worker.add_node("validate", RunnableLambda(validate))
    worker.add_edge(START, "generate")
    worker.add_edge("generate", "validate")
    worker.add_conditional_edges("validate", retry, {"generate": "generate", END: END})
    compiled_worker = worker.compile()

    def dispatch(state: State) -> list[Send]:
        chunks = state["chunks"]
        if not chunks:
            raise ValueError("No usable source chunks")
        count = state["request"].count
        weights = [len(chunk.text) for chunk in chunks]
        total = sum(weights)
        jobs = []
        for slot in range(count):
            target = (slot + 0.5) * total / count
            cumulative = 0
            chunk = chunks[-1]
            for candidate, weight in zip(chunks, weights, strict=True):
                cumulative += weight
                if cumulative >= target:
                    chunk = candidate
                    break
            jobs.append(
                Send(
                    "question",
                    Job(
                        slot=slot,
                        chunk=chunk,
                        request=state["request"],
                        attempts=0,
                        feedback="",
                        draft=None,
                        passed=False,
                    ),
                )
            )
        return jobs

    def question(job: Job) -> dict[str, list[Result]]:
        outcome = compiled_worker.invoke(job)
        draft = outcome["draft"]
        accepted = (
            MCQ(**draft.model_dump(), source=job["chunk"])
            if (outcome["passed"] and draft is not None)
            else None
        )
        return {
            "results": [
                Result(
                    slot=job["slot"],
                    question=accepted,
                    warning=""
                    if accepted
                    else f"Question {job['slot'] + 1}: {outcome['feedback']}",
                )
            ]
        }

    def finalize(state: State) -> dict[str, Quiz]:
        accepted: list[MCQ] = []
        warnings = []
        for result in sorted(state["results"], key=lambda item: item["slot"]):
            candidate = result["question"]
            if candidate is None:
                warnings.append(result["warning"])
                continue
            if any(
                ratio(candidate.question.casefold(), item.question.casefold()) >= 88
                for item in accepted
            ):
                warnings.append("A near-duplicate question was omitted")
                continue
            order = list(range(4))
            random.SystemRandom().shuffle(order)
            shuffled = candidate.model_copy(
                update={
                    "options": [candidate.options[index] for index in order],
                    "answer_index": order.index(candidate.answer_index),
                }
            )
            accepted.append(shuffled)
        if len(accepted) < state["request"].count:
            warnings.append(
                f"Returned {len(accepted)} of {state['request'].count} requested "
                "questions; rejected or duplicate questions are not included."
            )
        return {
            "quiz": Quiz(requested=state["request"].count, questions=accepted, warnings=warnings)
        }

    graph = StateGraph(State)
    graph.add_node("question", RunnableLambda(question))
    graph.add_node("finalize", finalize)
    graph.add_conditional_edges(START, dispatch, ["question"])
    graph.add_edge("question", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()
