import argparse
import html
import logging
import socket
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import gradio as gr

from mcqgen.config import Settings
from mcqgen.llm import Model, ModelConfigError
from mcqgen.schemas import MCQ, Difficulty, Quiz, Request
from mcqgen.service import generate_quiz

CSS = """
:root { --ink: #20352c; --accent: #137b59; --coral: #ce5548; }
.gradio-container { max-width: 1120px !important; font-family: 'IBM Plex Sans', sans-serif; }
body { background: #fafcfa; }
.app-title { border-bottom: 2px solid var(--ink); padding: 20px 0 18px; margin-bottom: 20px; }
.app-title h1 { font-family: 'IBM Plex Serif', Georgia, serif; font-size: 32px;
  line-height: 1.2; color: var(--ink); margin: 0; letter-spacing: 0; }
.app-title span { color: var(--coral); }
.quiz-heading { font-size: 20px; line-height: 1.4; color: var(--ink); margin: 16px 0 8px; }
.quiz-meta { font-size: 13px; color: #59665f; margin-bottom: 8px; }
.answer { border-left: 3px solid var(--accent); padding: 8px 16px; line-height: 1.6; }
.question-block { border-top: 1px solid #d8e2db; padding-top: 16px; }
button, input, .block { border-radius: 6px !important; }
button.primary { background: var(--accent) !important; border-color: var(--accent) !important; }
@media (max-width: 600px) { .app-title h1 { font-size: 28px; } }
"""
HEAD = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?'
    "family=IBM+Plex+Sans:wght@400;500;600&amp;"
    'family=IBM+Plex+Serif:wght@500&amp;display=swap" rel="stylesheet">'
)


def check_answer(selected: int | None, question: MCQ) -> str:
    if selected is None:
        return '<div class="answer">Select an answer first.</div>'
    correct = selected == question.answer_index
    label = "Correct" if correct else "Not quite"
    answer = html.escape(question.options[question.answer_index])
    return (
        f'<div class="answer"><strong>{label}. '
        f"{'ABCD'[question.answer_index]}. {answer}</strong>"
        f"<p>{html.escape(question.explanation)}</p>"
        f"<p><em>&quot;{html.escape(question.quote)}&quot;</em></p>"
        f"<small>{html.escape(question.source.document)} &middot; "
        f"Page {question.source.page}</small></div>"
    )


def create_app(
    settings: Settings | None = None,
    generator: Model | None = None,
    judge: Model | None = None,
) -> gr.Blocks:
    settings = settings or Settings()

    def run(
        files: list[str] | None,
        count: float,
        difficulty: str,
    ) -> Iterator[tuple[dict[str, Any] | None, str, str | None]]:
        yield None, "Reading documents...", None
        if not files:
            raise gr.Error("Upload at least one PDF")
        request = Request(count=int(count), difficulty=Difficulty(difficulty))
        try:
            for completed, quiz in generate_quiz(files, request, settings, generator, judge):
                if quiz is None:
                    yield None, f"Reviewed {completed} / {request.count} questions", None
                    continue
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".json",
                    prefix="quiz-",
                    encoding="utf-8",
                    delete=False,
                ) as file:
                    file.write(quiz.model_dump_json(indent=2))
                    path = file.name
                status = f"{len(quiz.questions)} questions ready"
                if quiz.warnings:
                    status += f". {request.count - len(quiz.questions)} omitted by quality checks."
                try:
                    yield quiz.model_dump(mode="json"), status, path
                finally:
                    Path(path).unlink(missing_ok=True)
        except (ModelConfigError, ValueError) as error:
            raise gr.Error(str(error)) from None
        except Exception:
            logging.getLogger(__name__).exception("Quiz generation failed (provider or PDF error)")
            raise gr.Error(
                "Generation failed. Check the PDF, provider credentials, model availability, "
                "and inference quota, then retry."
            ) from None

    with gr.Blocks(title="MCQ Studio", delete_cache=(3600, 3600)) as app:
        gr.HTML('<header class="app-title"><h1>MCQ <span>Studio</span></h1></header>')
        quiz_state = gr.State(None, time_to_live=3600)
        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                uploads = gr.File(
                    label="Documents", file_types=[".pdf"], file_count="multiple", type="filepath"
                )
                count = gr.Slider(1, 50, value=10, step=1, label="Questions")
                difficulty = gr.Radio(
                    ["easy", "medium", "hard"], value="medium", label="Difficulty"
                )
                generate = gr.Button("Generate", variant="primary")
                status = gr.Textbox(label="Status", value="Ready", interactive=False)
                download = gr.File(label="Download quiz", interactive=False)
            with gr.Column(scale=2, min_width=300):
                gr.HTML('<h2 class="quiz-heading">Your Quiz</h2>')

                @gr.render(inputs=quiz_state)
                def render_quiz(data: dict[str, Any] | None) -> None:
                    if data is None:
                        return
                    quiz = Quiz.model_validate(data)
                    for index, question in enumerate(quiz.questions, start=1):
                        gr.HTML(
                            f'<section class="question-block"><div class="quiz-meta">'
                            f"{index:02d} / {len(quiz.questions):02d} &middot; "
                            f"{question.difficulty.value.title()}</div>"
                            f'<h3 class="quiz-heading">{html.escape(question.question)}</h3>'
                            "</section>"
                        )
                        choices = gr.Radio(
                            [
                                (f"{'ABCD'[option_index]}. {option}", option_index)
                                for option_index, option in enumerate(question.options)
                            ],
                            label=f"Answer {index}",
                        )
                        check = gr.Button("Check answer", size="sm")
                        feedback = gr.HTML()

                        def reveal(selected: int | None, item: MCQ = question) -> str:
                            return check_answer(selected, item)

                        check.click(
                            reveal, inputs=choices, outputs=feedback, api_visibility="private"
                        )

        generate.click(
            run,
            inputs=[uploads, count, difficulty],
            outputs=[quiz_state, status, download],
            concurrency_limit=1,
            api_name="generate",
        )
    return app.queue(max_size=8)


def available_port(host: str, start: int) -> int:
    for port in range(start, min(start + 20, 65536)):
        with socket.socket() as probe:
            try:
                probe.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError("No available application port")


def main() -> None:
    parser = argparse.ArgumentParser(prog="mcqgen")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("serve")
    generate = commands.add_parser("generate")
    generate.add_argument("pdf", nargs="+")
    generate.add_argument("-n", "--count", type=int, default=10)
    generate.add_argument(
        "--difficulty", choices=[item.value for item in Difficulty], default="medium"
    )
    generate.add_argument("-o", "--output", default="quiz.json")
    args = parser.parse_args()
    try:
        settings = Settings()
    except ValueError:
        parser.exit(1, "Invalid configuration; check .env settings and numeric limits.\n")
    if args.command == "generate":
        try:
            request = Request(count=args.count, difficulty=Difficulty(args.difficulty))
            for _, quiz in generate_quiz(args.pdf, request, settings):
                if quiz:
                    Path(args.output).write_text(quiz.model_dump_json(indent=2), encoding="utf-8")
                    print(f"Saved {len(quiz.questions)} questions to {args.output}")
                    for warning in quiz.warnings:
                        print(warning)
        except (ModelConfigError, ValueError, OSError) as error:
            parser.exit(1, f"Generation failed: {error}\n")
        return
    app = create_app(settings)
    app.launch(
        server_name=settings.app_host,
        server_port=available_port(settings.app_host, settings.app_port),
        share=False,
        max_file_size="25mb",
        footer_links=[],
        css=CSS,
        head=HEAD,
        theme=gr.themes.Base(primary_hue="emerald", secondary_hue="rose"),
    )


if __name__ == "__main__":
    main()
