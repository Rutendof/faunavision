# FaunaVision inference API for Hugging Face Spaces (Docker SDK, CPU).
#
# Expects an exported model in the build context (see scripts/export_onnx.py):
#   exports/model.onnx + exports/model.json
# Build and run locally:
#   docker build -t faunavision-api .
#   docker run --rm -p 7860:7860 faunavision-api
# Image contains no PyTorch and no training data: ONNX Runtime + FastAPI only.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Spaces run the container as uid 1000; match it so files are owned correctly
RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/faunavision

# Dependencies first so code changes don't reinstall them
COPY --chown=app pyproject.toml ./
COPY --chown=app src ./src
RUN pip install ".[api]"

COPY --chown=app exports/model.onnx exports/model.json ./model/

USER app
ENV FAUNAVISION_MODEL=/home/app/faunavision/model/model.onnx
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4)"

CMD ["uvicorn", "faunavision.api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
