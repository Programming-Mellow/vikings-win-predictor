# One image for every service: API, Streamlit app, weekly job, and bootstrap.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    VIKPRED_HOME=/app

# XGBoost needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies before copying the rest so code changes don't bust this layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY app ./app
COPY reports ./reports

# Run as a non-root user; data/, models/ and mlruns/ are mounted volumes.
RUN useradd --create-home --uid 1000 vikpred \
    && mkdir -p data models mlruns \
    && chown -R vikpred /app
USER vikpred

EXPOSE 8000 8501
CMD ["uvicorn", "vikpred.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
