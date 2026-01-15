FROM python:3.13-slim

RUN pip install --no-cache-dir poetry

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.in-project true \
    && poetry config virtualenvs.create true \
    && poetry install --no-root --no-interaction --no-ansi

COPY . .

CMD ["/app/.venv/bin/python", "src/main.py"]