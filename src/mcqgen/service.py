from collections.abc import Iterator

from langsmith import Client, tracing_context

from mcqgen.config import Settings
from mcqgen.graph import build_graph
from mcqgen.ingestion import load_documents
from mcqgen.llm import Model, get_model
from mcqgen.schemas import Quiz, Request


def generate_quiz(
    paths: list[str],
    request: Request,
    settings: Settings,
    generator: Model | None = None,
    judge: Model | None = None,
) -> Iterator[tuple[int, Quiz | None]]:
    chunks = load_documents(paths)
    generator = generator or get_model(settings)
    judge = judge or get_model(settings, judge=True)
    graph = build_graph(generator, judge, settings.max_retries)
    completed = 0
    yield completed, None
    client = None
    if settings.langsmith_tracing:
        if not settings.langsmith_api_key:
            raise ValueError("Set LANGSMITH_API_KEY before enabling tracing")
        client = Client(api_key=settings.langsmith_api_key.get_secret_value())
    with tracing_context(
        enabled=settings.langsmith_tracing, client=client, project_name=settings.langsmith_project
    ):
        for update in graph.stream(
            {"chunks": chunks, "request": request, "results": []},
            config={"max_concurrency": settings.max_concurrency},
            stream_mode="updates",
        ):
            if "question" in update:
                completed += len(update["question"]["results"])
                yield completed, None
            if "finalize" in update:
                yield completed, update["finalize"]["quiz"]
