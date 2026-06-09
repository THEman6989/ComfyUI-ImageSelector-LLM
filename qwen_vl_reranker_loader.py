"""
Qwen3-VL-Reranker Loader — standalone model loader for reranking.

Loads the Qwen3-VL-Reranker model and outputs it as a reusable reference.
Connect to BeatDropSelectorEmbeddingNode.reranker_model input.

Supports: fp16, bf16, int8, fp32
Devices: cuda:0, cuda:1, cuda:2, cuda:3, cpu, auto

Place in: ComfyUI-ImageSelector-LLM/qwen_vl_reranker_loader.py
"""

import torch
from pathlib import Path

# Shared cache
_RERANKER_CACHE = {}
_ALLOWED_RERANKER_MODELS = {"Qwen/Qwen3-VL-Reranker-8B"}


def _validate_model_path(model_path):
    model_path = str(model_path or "").strip()
    if model_path in _ALLOWED_RERANKER_MODELS:
        return model_path
    p = Path(model_path).expanduser()
    if p.exists():
        return str(p.resolve())
    raise ValueError(
        f"Refusing to load untrusted remote reranker model_path={model_path!r}. "
        "Use the built-in Qwen/Qwen3-VL-Reranker-8B or a local path."
    )


def _get_reranker_model(model_path, device, dtype_str):
    """Lazy-load and cache Qwen3-VL-Reranker model."""
    model_path = _validate_model_path(model_path)
    cache_key = (model_path, device, dtype_str)
    if cache_key in _RERANKER_CACHE:
        return _RERANKER_CACHE[cache_key]

    print(f"[RerankerLoader] Loading: {model_path} | device={device} | dtype={dtype_str}")

    torch_dtype_map = {
        "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32,
    }
    load_kwargs = {"trust_remote_code": True, "local_files_only": False}

    if dtype_str == "int8":
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = "auto" if device in ("auto", "cuda", "cuda:0") else device
    elif dtype_str in torch_dtype_map:
        load_kwargs["torch_dtype"] = torch_dtype_map[dtype_str]
    else:
        load_kwargs["torch_dtype"] = torch.float16

    if "device_map" not in load_kwargs:
        load_kwargs["device_map"] = device if device != "auto" else "auto"

    # Try official Qwen3VLReranker wrapper
    try:
        from src.models.qwen3_vl_reranker import Qwen3VLReranker
        model = Qwen3VLReranker(model_name_or_path=model_path, **load_kwargs)
        model_type = "qwen3vl"
    except ImportError:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_path, **load_kwargs)
        model.eval()
        model_type = "transformers"

    _RERANKER_CACHE[cache_key] = (model, model_type)
    print(f"[RerankerLoader] Model loaded (type={model_type})")
    return _RERANKER_CACHE[cache_key]


class QwenVLRerankerLoader:
    """Loads Qwen3-VL-Reranker model for downstream nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "Qwen/Qwen3-VL-Reranker-8B",
                    "multiline": False,
                    "tooltip": "HF model ID or local path.",
                }),
                "device": (["auto", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cpu"], {
                    "default": "auto",
                    "tooltip": "CUDA device. Can differ from embedding device.",
                }),
                "dtype": (["fp16", "bf16", "int8", "fp32"], {
                    "default": "fp16",
                    "tooltip": "fp16: ~16GB. int8: ~8GB.",
                }),
                "keep_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in VRAM. OFF = unload after use.",
                }),
            },
        }

    RETURN_TYPES = ("QWEN_VL_RERANKER", "STRING")
    RETURN_NAMES = ("model", "model_info")
    FUNCTION = "load_model"
    CATEGORY = "Amin/Beatdrop"

    def load_model(self, model_path, device, dtype, keep_loaded):
        model_path = _validate_model_path(model_path)
        model, model_type = _get_reranker_model(model_path, device, dtype)
        cache_key = (model_path, device, dtype)
        info = str({"model_path": model_path, "device": device, "dtype": dtype, "type": model_type})
        return (cache_key, info)


class QwenVLRerankerUnloader:
    """Explicitly unload the Qwen3-VL-Reranker model from VRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN_VL_RERANKER", {"tooltip": "Model reference from QwenVLRerankerLoader."}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "unload"
    CATEGORY = "Amin/Beatdrop"
    OUTPUT_NODE = True

    def unload(self, model):
        global _RERANKER_CACHE
        cache_key = model
        if cache_key in _RERANKER_CACHE:
            m, _ = _RERANKER_CACHE.pop(cache_key)
            del m
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[RerankerUnloader] Model unloaded")
        return ()


NODE_CLASS_MAPPINGS = {
    "QwenVLRerankerLoader": QwenVLRerankerLoader,
    "QwenVLRerankerUnloader": QwenVLRerankerUnloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVLRerankerLoader": "🧠 Qwen VL Reranker Loader",
    "QwenVLRerankerUnloader": "🗑 Qwen VL Reranker Unloader",
}
