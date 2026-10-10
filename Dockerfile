FROM python:3.13-slim
WORKDIR /app
RUN useradd --create-home app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH" APP_HOST=0.0.0.0 APP_PORT=7860
USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=4)"
CMD ["mcqgen", "serve"]