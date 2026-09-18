import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import model_cache


@pytest.fixture
def cache(monkeypatch, tmp_path):
    import huggingface_hub

    import laya

    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LAYA_DEVICE", "cpu")
    model_dir = tmp_path / "cache" / model_cache.MODEL_REVISION
    downloads = []

    def download(**kwargs):
        downloads.append(kwargs)
        for name in model_cache.REQUIRED_FILES:
            path = Path(kwargs["local_dir"]) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("complete")
        return str(model_dir)

    def load(path, **kwargs):
        assert all(
            (Path(path) / name).read_text() == "complete"
            for name in model_cache.REQUIRED_FILES
        )
        return SimpleNamespace(device=SimpleNamespace(type="cpu"))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(laya, "load", load)
    return model_dir, downloads


def test_ready_cache_loads_when_downloads_are_unavailable(monkeypatch, cache):
    import huggingface_hub

    model_dir, downloads = cache
    model_cache.load_agent()
    assert (model_dir / ".ready").is_file()

    def offline(**kwargs):
        raise ConnectionError("No network")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", offline)
    agent = model_cache.load_agent()
    assert agent.device.type == "cpu"
    assert len(downloads) == 1


def test_failed_load_is_not_marked_ready_and_can_resume(monkeypatch, cache):
    import laya

    model_dir, _ = cache
    original = laya.load

    def fail(*args, **kwargs):
        raise RuntimeError("incompatible weights")

    monkeypatch.setattr(laya, "load", fail)
    with pytest.raises(RuntimeError, match="incompatible"):
        model_cache.load_agent()
    assert not (model_dir / ".ready").exists()
    assert (model_dir / "model.safetensors").is_file()
    monkeypatch.setattr(laya, "load", original)
    model_cache.load_agent()
    assert (model_dir / ".ready").is_file()


def test_partial_cache_repair_failure_removes_stale_marker(monkeypatch, cache):
    import huggingface_hub

    model_dir, _ = cache
    model_cache.load_agent()
    (model_dir / "encoder/config.json").unlink()

    def interrupted(**kwargs):
        raise ConnectionError("interrupted")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", interrupted)
    with pytest.raises(ConnectionError):
        model_cache.load_agent()
    assert not (model_dir / ".ready").exists()


def test_corrupt_ready_cache_fails_without_redownloading(monkeypatch, cache):
    import laya

    _, downloads = cache
    model_cache.load_agent()

    def corrupt(*args, **kwargs):
        raise ValueError("corrupt model")

    monkeypatch.setattr(laya, "load", corrupt)
    with pytest.raises(ValueError, match="corrupt"):
        model_cache.load_agent()
    assert len(downloads) == 1


def test_concurrent_initializers_cannot_load_partial_snapshot(monkeypatch, cache):
    import huggingface_hub

    import laya

    model_dir, downloads = cache
    entered = threading.Event()
    release = threading.Event()
    loaded = threading.Event()
    original_download = huggingface_hub.snapshot_download
    original_load = laya.load

    def slow_download(**kwargs):
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.safetensors").write_text("partial")
        entered.set()
        assert release.wait(5)
        return original_download(**kwargs)

    def load(*args, **kwargs):
        agent = original_load(*args, **kwargs)
        loaded.set()
        return agent

    monkeypatch.setattr(huggingface_hub, "snapshot_download", slow_download)
    monkeypatch.setattr(laya, "load", load)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(model_cache.load_agent)
        second = None
        try:
            assert entered.wait(2)
            second = pool.submit(model_cache.load_agent)
            assert not loaded.wait(0.1)
        finally:
            release.set()
        assert first.result(timeout=3).device.type == "cpu"
        assert second.result(timeout=3).device.type == "cpu"
    assert len(downloads) == 1
    assert (model_dir / ".ready").is_file()


def test_missing_cuda_fails_before_downloading(monkeypatch, cache):
    import torch

    _, downloads = cache
    monkeypatch.setenv("LAYA_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        model_cache.load_agent()
    assert not downloads


def test_cuda_cpu_fallback_is_not_marked_ready(monkeypatch, cache):
    import torch

    model_dir, _ = cache
    monkeypatch.setenv("LAYA_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="did not load onto CUDA"):
        model_cache.load_agent()
    assert not (model_dir / ".ready").exists()


def test_missing_runpod_mount_fails_without_creating_cache(monkeypatch, cache):
    _, downloads = cache
    monkeypatch.setenv("MODEL_CACHE_DIR", "/runpod-volume/models/laya")
    monkeypatch.setattr(Path, "is_mount", lambda path: False)
    with pytest.raises(RuntimeError, match="network volume"):
        model_cache.load_agent()
    assert not downloads
