FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

# Install dependencies first (better layer caching)
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

# Copy application code and model
COPY app ./app
COPY model ./model

# Place the virtualenv on PATH so uvicorn runs directly without uv needing network at startup
ENV PATH="/code/.venv/bin:$PATH"

WORKDIR /code

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


