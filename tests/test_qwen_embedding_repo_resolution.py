import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO.parents[1]
for path in (str(COMFY_ROOT), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

loader = importlib.import_module("qwen_vl_embedding_loader")


def test_embedding_repo_can_be_resolved_from_environment(monkeypatch, tmp_path):
    repo = tmp_path / "Qwen3-VL-Embedding"
    wrapper = repo / "src" / "models" / "qwen3_vl_embedding.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("# test wrapper\n", encoding="utf-8")
    monkeypatch.setenv("QWEN3_VL_EMBEDDING_REPO", str(repo))

    assert loader._resolve_embedding_repo() == repo.resolve()


def test_loader_honors_requested_cuda_device(monkeypatch, tmp_path):
    repo = tmp_path / "Qwen3-VL-Embedding"
    wrapper = repo / "src" / "models" / "qwen3_vl_embedding.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("# test wrapper\n", encoding="utf-8")
    monkeypatch.setattr(loader, "_resolve_embedding_repo", lambda: repo)

    entered = []
    state = {"device": None}

    @contextmanager
    def fake_cuda_device(device):
        entered.append(device)
        previous = state["device"]
        state["device"] = device
        try:
            yield
        finally:
            state["device"] = previous

    class FakeEmbedder:
        def __init__(self, **kwargs):
            self.loaded_on = state["device"]

    fake_module = types.ModuleType("src.models.qwen3_vl_embedding")
    setattr(fake_module, "Qwen3VLEmbedder", FakeEmbedder)
    monkeypatch.setitem(sys.modules, "src.models.qwen3_vl_embedding", fake_module)
    monkeypatch.setattr(loader.torch.cuda, "device", fake_cuda_device)
    loader._QWEN_EMBEDDING_CACHE.clear()

    model, processor, dim = loader._get_qwen_embedding_model(
        "Qwen/Qwen3-VL-Embedding-8B", "cuda:1", "fp16"
    )

    assert entered == ["cuda:1"]
    assert model.loaded_on == "cuda:1"
    assert processor == "qwen3vl_embedder"
    assert dim == 4096
