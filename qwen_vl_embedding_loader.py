"""
Qwen3-VL-Embedding Loader — standalone model loader node.

Loads the Qwen3-VL-Embedding model once and outputs it as a reusable reference.
Connect to BeatDropSelectorEmbeddingNode or any other node that needs
image/text embeddings.

Supports: fp16, bf16, int8, fp8, fp32
Devices: cuda:0, cuda:1, cuda:2, cuda:3, cpu, auto

Place in: ComfyUI-ImageSelector-LLM/qwen_vl_embedding_loader.py
"""

import os
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image

# Shared cache — same as beatdrop_selector_embedding.py
_QWEN_EMBEDDING_CACHE = {}
_ALLOWED_EMBEDDING_MODELS = {"Qwen/Qwen3-VL-Embedding-8B"}


def _resolve_embedding_repo():
    """Resolve the trusted Qwen wrapper repo from durable then legacy locations."""
    candidates = []
    configured = str(os.environ.get("QWEN3_VL_EMBEDDING_REPO", "")).strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend([
        Path.home() / "experi" / "krams" / "Qwen3-VL-Embedding",
        Path("/tmp/Qwen3-VL-Embedding"),
    ])
    checked = []
    for candidate in candidates:
        repo = candidate.resolve()
        expected = repo / "src" / "models" / "qwen3_vl_embedding.py"
        checked.append(str(expected))
        if expected.is_file():
            return repo
    raise RuntimeError(
        "Required Qwen3-VL-Embedding wrapper not found. Checked: "
        + ", ".join(checked)
    )


def _validate_model_path(model_path):
    model_path = str(model_path or "").strip()
    if model_path in _ALLOWED_EMBEDDING_MODELS:
        return model_path
    p = Path(model_path).expanduser()
    if p.exists():
        return str(p.resolve())
    raise ValueError(
        f"Refusing to load untrusted remote embedding model_path={model_path!r}. "
        "Use the built-in Qwen/Qwen3-VL-Embedding-8B or a local path."
    )


def _get_qwen_embedding_model(model_path, device, dtype_str):
    """Lazy-load and cache Qwen3-VL-Embedding model via Qwen3VLEmbedder."""
    model_path = _validate_model_path(model_path)
    cache_key = (model_path, device, dtype_str)
    if cache_key in _QWEN_EMBEDDING_CACHE:
        return _QWEN_EMBEDDING_CACHE[cache_key]

    print(f"[QwenLoader] Loading: {model_path} | device={device} | dtype={dtype_str}")

    import sys as _sys
    _embed_repo = _resolve_embedding_repo()
    if str(_embed_repo) not in _sys.path:
        _sys.path.insert(0, str(_embed_repo))

    torch_dtype = torch.float16 if dtype_str == "fp16" else torch.float32

    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
    load_context = (
        torch.cuda.device(device)
        if str(device).startswith("cuda:") and torch.cuda.is_available()
        else nullcontext()
    )
    with load_context:
        model = Qwen3VLEmbedder(
            model_name_or_path=model_path,
            torch_dtype=torch_dtype,
        )
    if str(device) == "cpu" and hasattr(model, "model"):
        model.model.to("cpu")
    dim = 4096

    _QWEN_EMBEDDING_CACHE[cache_key] = (model, "qwen3vl_embedder", dim)
    print(f"[QwenLoader] Loaded via Qwen3VLEmbedder (dim={dim})")
    return _QWEN_EMBEDDING_CACHE[cache_key]


class QwenVLEmbeddingLoader:
    """Loads Qwen3-VL-Embedding model and outputs it for downstream nodes.

    The output is a tuple of (model_ref, processor, dim) serialized as a
    dictionary reference. Connect to any node that needs embeddings.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "Qwen/Qwen3-VL-Embedding-8B",
                    "multiline": False,
                    "tooltip": "HF model ID or local path.",
                }),
                "device": (["auto", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cpu"], {
                    "default": "auto",
                    "tooltip": "CUDA device. auto = best available.",
                }),
                "dtype": (["fp16", "bf16", "int8", "fp8", "fp32"], {
                    "default": "fp16",
                    "tooltip": "fp16: ~16GB. int8/fp8: ~8GB.",
                }),
                "keep_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in VRAM after loading. OFF = unload after downstream use.",
                }),
            },
        }

    RETURN_TYPES = ("QWEN_VL_EMBEDDING", "INT", "STRING")
    RETURN_NAMES = ("model", "dim", "model_info")
    FUNCTION = "load_model"
    CATEGORY = "Amin/Beatdrop"

    def load_model(self, model_path, device, dtype, keep_loaded):
        model_path = _validate_model_path(model_path)
        model, processor, dim = _get_qwen_embedding_model(model_path, device, dtype)

        # Package model reference — we can't pass Python objects through
        # ComfyUI wires, so we use the global cache key as reference
        cache_key = (model_path, device, dtype)

        info = {
            "model_path": model_path,
            "device": device,
            "dtype": dtype,
            "dim": dim,
            "cache_key": str(cache_key),
            "keep_loaded": keep_loaded,
        }

        return (cache_key, dim, str(info))


class QwenVLEmbeddingUnloader:
    """Explicitly unload the Qwen3-VL-Embedding model from VRAM.

    Connect after all embedding-dependent nodes have finished.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN_VL_EMBEDDING", {"tooltip": "Model reference from QwenVLEmbeddingLoader."}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "unload"
    CATEGORY = "Amin/Beatdrop"
    OUTPUT_NODE = True

    def unload(self, model):
        global _QWEN_EMBEDDING_CACHE
        cache_key = model  # model IS the cache_key tuple
        if cache_key in _QWEN_EMBEDDING_CACHE:
            m, p, d = _QWEN_EMBEDDING_CACHE.pop(cache_key)
            del m, p
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[QwenUnloader] Model unloaded from VRAM")
        return ()


# ── Node registration ──────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "QwenVLEmbeddingLoader": QwenVLEmbeddingLoader,
    "QwenVLEmbeddingUnloader": QwenVLEmbeddingUnloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVLEmbeddingLoader": "🧠 Qwen VL Embedding Loader",
    "QwenVLEmbeddingUnloader": "🗑 Qwen VL Embedding Unloader",
}
