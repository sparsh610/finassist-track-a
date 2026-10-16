# FastAPI gateway image: retrieval + prompt building + JSON validation.
# Generation happens in the separate vLLM container.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU torch keeps the image small; embeddings + reranker run on CPU fine at this corpus size.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
 && pip install sentence-transformers==3.3.1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml .
COPY configs ./configs
COPY src ./src
RUN pip install -e .

# Pre-download embedding + reranker weights so the container starts offline.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"

RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
CMD ["uvicorn", "finassist.serve.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
