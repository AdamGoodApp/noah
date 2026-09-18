FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PORT_HEALTH=8000 \
    HEALTH_CHECK_PATH=/ping \
    MODEL_CACHE_DIR=/runpod-volume/models/laya \
    HF_HOME=/runpod-volume/huggingface \
    LAYA_DEVICE=cuda

# ModernBERT's CUDA path uses Triton to compile its host-side driver.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py README.md LICENSE requirements-server.txt ./
COPY laya/ ./laya/
RUN python -m pip install --no-cache-dir -r requirements-server.txt .

COPY api.py api_models.py model_cache.py ./

EXPOSE 8000
CMD ["python", "api.py"]
