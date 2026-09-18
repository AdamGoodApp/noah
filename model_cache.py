"""Revision-pinned Laya loading backed by a persistent network volume."""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from filelock import FileLock

if TYPE_CHECKING:
    import laya

logger = logging.getLogger(__name__)
MODEL_ID = "convaiinnovations/laya"
MODEL_REVISION = "7c76b622dfc5cac71b2dc1c29873efe2ce509a05"
REQUIRED_FILES = (
    "model.safetensors",
    "rl_agent_config.json",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)


def load_agent() -> "laya.Agent":
    # Hugging Face reads cache environment variables on import. The application
    # configures its environment before entering this function.
    import torch
    from huggingface_hub import snapshot_download

    import laya

    device = os.environ.get("LAYA_DEVICE", "cuda")
    if device not in {"cuda", "cpu", "mps"}:
        raise ValueError("LAYA_DEVICE must be cuda, cpu, or mps")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")

    cache = Path(os.environ.get("MODEL_CACHE_DIR", ".model-cache")).absolute()
    volume = Path("/runpod-volume")
    if cache.is_relative_to(volume) and (
        not volume.is_mount() or not os.access(volume, os.W_OK)
    ):
        raise RuntimeError(
            "A writable network volume must be mounted at /runpod-volume"
        )
    cache.mkdir(parents=True, exist_ok=True)

    with FileLock(cache / ".initialize.lock"):
        model_dir = cache / MODEL_REVISION
        ready = model_dir / ".ready"
        cached = ready.is_file() and all(
            (model_dir / filename).is_file() for filename in REQUIRED_FILES
        )
        if cached:
            logger.info("Model cache hit: %s revision %s", MODEL_ID, MODEL_REVISION)
        else:
            # A stale marker must not bless a failed repair of a partial cache.
            ready.unlink(missing_ok=True)
            logger.info(
                "Initializing model cache: %s revision %s", MODEL_ID, MODEL_REVISION
            )
            snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                local_dir=str(model_dir),
                allow_patterns=[
                    "model.safetensors",
                    "rl_agent_config.json",
                    "encoder/*",
                    "tokenizer/*",
                ],
                token=os.environ.get("HF_TOKEN") or None,
            )
            if not all((model_dir / filename).is_file() for filename in REQUIRED_FILES):
                raise RuntimeError("Model snapshot is incomplete")

        agent = laya.load(
            str(model_dir), device=device, token=os.environ.get("HF_TOKEN") or None
        )
        if device == "cuda" and agent.device.type != "cuda":
            raise RuntimeError("Model did not load onto CUDA")

        if not cached:
            marker = model_dir / ".ready.tmp"
            marker.write_text(MODEL_REVISION + "\n", encoding="utf-8")
            os.replace(marker, ready)

    logger.info("Model loaded on %s", agent.device.type)
    if agent.device.type == "cuda":
        logger.info("CUDA GPU: %s", torch.cuda.get_device_name(agent.device))
    return agent
