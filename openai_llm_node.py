import base64
import io
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_SELECTOR_SYSTEM_PROMPT = (
    "You are a visual matching judge. Compare the reference person/video to the "
    "numbered candidate images. Focus on clothing, colors, patterns, silhouette, "
    "visible accessories, and overall outfit. Ignore background, lighting, pose, "
    "camera angle, and image quality. Return only valid JSON."
)
SUBDIRECTORY_SELECTOR_SYSTEM_PROMPT = (
    "You choose which candidate image folders should be searched before visual "
    "matching. Return only valid JSON."
)
SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def _tensor_to_numpy(image_tensor):
    """Convert a ComfyUI tensor or numpy-like image value to a numpy array."""
    if hasattr(image_tensor, "detach"):
        return image_tensor.detach().cpu().numpy()
    return np.asarray(image_tensor)


def _to_uint8_image_array(image_array):
    """Clamp image data and convert one image in HWC format to uint8."""
    image_array = np.asarray(image_array)

    if image_array.ndim != 3:
        raise ValueError(f"Expected image with shape [H,W,C], got {image_array.shape}")

    if image_array.shape[-1] not in (1, 3, 4):
        raise ValueError(
            "Expected channel-last ComfyUI image data with 1, 3, or 4 channels, "
            f"got shape {image_array.shape}"
        )

    if np.issubdtype(image_array.dtype, np.floating):
        image_array = np.clip(image_array, 0.0, 1.0)
        image_array = np.rint(image_array * 255.0).astype(np.uint8)
    else:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    return image_array


def _image_array_to_pil(image_array):
    """Convert one uint8 HWC image to an RGB PIL image for drawing/encoding."""
    image_array = _to_uint8_image_array(image_array)

    if image_array.shape[-1] == 1:
        return Image.fromarray(image_array[:, :, 0], "L").convert("RGB")
    if image_array.shape[-1] == 4:
        return Image.fromarray(image_array, "RGBA").convert("RGB")
    return Image.fromarray(image_array, "RGB")


def _image_batch_to_pil_images(image_tensor):
    """
    ComfyUI IMAGE values are normally [B,H,W,C] float tensors in the 0-1 range.
    A single image may appear as [H,W,C]. Batch-aware nodes must keep B intact
    instead of using image_tensor[0], because each batch item is a candidate.
    """
    image_array = _tensor_to_numpy(image_tensor)

    if image_array.ndim == 3:
        image_array = image_array[None, ...]
    elif image_array.ndim != 4:
        raise ValueError(
            "Expected ComfyUI IMAGE tensor with shape [B,H,W,C] or [H,W,C], "
            f"got {image_array.shape}"
        )

    return [_image_array_to_pil(image_array[index]) for index in range(image_array.shape[0])]


def _pil_image_to_comfy_image(image):
    """Convert a PIL image to a ComfyUI IMAGE batch shaped [1,H,W,C]."""
    return _pil_list_to_comfy_batch([image])


def _pil_list_to_comfy_batch(images):
    """Convert a list of PIL images into a ComfyUI IMAGE batch [B,H,W,C]."""
    if not images:
        return None

    # Use the first image's size as the reference to ensure a valid batch
    width, height = images[0].size

    processed_images = []
    for img in images:
        if img.size != (width, height):
            img = img.resize((width, height), Image.Resampling.LANCZOS)
        img_array = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        processed_images.append(img_array)

    batch_array = np.stack(processed_images, axis=0)

    try:
        import torch

        return torch.from_numpy(batch_array)
    except Exception:
        return batch_array


def _normalize_filter_text(value):
    return re.sub(r"[_\W]+", " ", str(value or "").casefold()).strip()


def _filter_tokens(value):
    return [
        token
        for token in _normalize_filter_text(value).split()
        if len(token) >= 3
    ]


def _split_subdirectory_filter(value):
    return [
        _normalize_filter_text(part)
        for part in str(value or "").split(",")
        if _normalize_filter_text(part)
    ]


def _subdirectory_matches_filter(path, root, filters):
    relative_text = _normalize_filter_text(path.relative_to(root).as_posix())
    name_text = _normalize_filter_text(path.name)

    return any(
        filter_text in relative_text
        or filter_text in name_text
        or relative_text in filter_text
        or name_text in filter_text
        for filter_text in filters
    )


def _subdirectory_matches_prompt(path, prompt):
    folder_tokens = _filter_tokens(path.name)
    if not folder_tokens:
        return False

    prompt_text = _normalize_filter_text(prompt)
    prompt_tokens = _filter_tokens(prompt)
    folder_text = _normalize_filter_text(path.name)

    return any(token in prompt_text for token in folder_tokens) or any(
        token in folder_text for token in prompt_tokens
    )


def _select_subdirectories(root, subdirectory_filter, prompt):
    subdirectory_filter = str(subdirectory_filter or "").strip()
    if not subdirectory_filter:
        return [], "none"

    subdirectories = sorted(path for path in root.rglob("*") if path.is_dir())
    if not subdirectories:
        return [], "none"

    if subdirectory_filter.casefold() == "auto":
        selected = [
            path
            for path in subdirectories
            if _subdirectory_matches_prompt(path, prompt)
        ]
        return selected, "auto" if selected else "auto_no_match"

    filters = _split_subdirectory_filter(subdirectory_filter)
    selected = [
        path
        for path in subdirectories
        if _subdirectory_matches_filter(path, root, filters)
    ]
    return selected, "manual"


def _load_images_from_directory(directory, recursive=False, subdirectory_filter="", prompt=""):
    """Load every supported image file from a directory as RGB PIL images."""
    directory = str(directory or "").strip()
    if not directory:
        return [], []

    root = Path(directory).expanduser()
    if not root.exists():
        raise ValueError(f"candidate_directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"candidate_directory is not a directory: {root}")

    selected_subdirectories, filter_mode = _select_subdirectories(
        root,
        subdirectory_filter,
        prompt,
    )
    if selected_subdirectories:
        iterators = [
            subdirectory.rglob("*") if recursive else subdirectory.iterdir()
            for subdirectory in selected_subdirectories
        ]
        image_paths = sorted({
            path
            for iterator in iterators
            for path in iterator
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        })
    else:
        iterator = root.rglob("*") if recursive else root.iterdir()
        image_paths = sorted(
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )

    if filter_mode == "manual" and not selected_subdirectories:
        raise ValueError(
            "candidate_subdirectories did not match any subfolders in candidate_directory"
        )

    if selected_subdirectories and not image_paths:
        raise ValueError(
            "candidate_subdirectories matched subfolders, but no supported image files were found"
        )

    images = []
    loaded_paths = []
    errors = []
    for path in image_paths:
        try:
            with Image.open(path) as image:
                images.append(ImageOps.exif_transpose(image).convert("RGB").copy())
            loaded_paths.append(str(path))
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    if image_paths and not images:
        raise ValueError(
            "candidate_directory contains image files, but none could be loaded: "
            + "; ".join(errors[:5])
        )

    return images, loaded_paths


def _load_label_font():
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _draw_label(image, label):
    labelled = image.copy()
    draw = ImageDraw.Draw(labelled)
    font = _load_label_font()
    padding = 5

    try:
        bbox = draw.textbbox((0, 0), str(label), font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except Exception:
        text_width = max(8, len(str(label)) * 7)
        text_height = 12

    rect = (
        0,
        0,
        text_width + padding * 2,
        text_height + padding * 2,
    )
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.text((padding, padding), str(label), fill=(255, 255, 255), font=font)
    return labelled


def _build_contact_sheet(images, columns, labels=None):
    if not images:
        raise ValueError("Cannot build a contact sheet without images")

    columns = max(1, min(int(columns), len(images)))
    rows = int(math.ceil(len(images) / columns))
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)

    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), (255, 255, 255))

    for index, image in enumerate(images):
        cell_image = image
        if labels is not None:
            cell_image = _draw_label(cell_image, labels[index])

        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(cell_image, (x, y))

    return sheet


def _encode_pil_to_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


def _image_url_part(data_url, detail=None):
    image_url = {"url": data_url}
    if detail:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def _extract_choice_content(result):
    choices = result.get("choices") or []
    if not choices:
        raise ValueError("No response choices found")

    choice = choices[0]
    if "message" in choice:
        content = choice["message"].get("content", "")
    else:
        content = choice.get("text", "")

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts)

    return str(content)


class OpenAILLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.",
                    "placeholder": "Enter your prompt here..."
                }),
                "endpoint": ("STRING", {
                    "multiline": False,
                    "default": "https://api.openai.com/v1/chat/completions",
                    "placeholder": "OpenAI-compatible endpoint URL"
                }),
                "api_token": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Your API token"
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "model": ("STRING", {
                    "multiline": False,
                    "default": "gpt-4-vision-preview",
                    "placeholder": "Model name"
                }),
                "max_tokens": ("INT", {
                    "default": 150,
                    "min": 1,
                    "max": 4096,
                    "step": 1
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "image_detail": (["omit", "low", "high", "auto"], {
                    "default": "auto"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "LLM"

    def _encode_image_to_base64(self, image_tensor):
        """Convert a ComfyUI IMAGE tensor to a PNG data URL."""
        try:
            images = _image_batch_to_pil_images(image_tensor)
            # Older code used image_tensor[0]. If a batch is supplied here, send a
            # contact sheet so no batch item is silently dropped.
            image = images[0] if len(images) == 1 else _build_contact_sheet(images, 4)
            return _encode_pil_to_data_url(image)
        except Exception as e:
            raise Exception(f"Failed to encode image: {str(e)}")

    def generate_text(
        self,
        prompt,
        endpoint,
        api_token,
        model="gpt-4-vision-preview",
        max_tokens=150,
        temperature=0.7,
        image=None,
        image_detail="auto",
    ):
        try:
            headers = {"Content-Type": "application/json"}
            if api_token:
                headers["Authorization"] = f"Bearer {api_token}"

            if image is not None:
                image_data_url = self._encode_image_to_base64(image)
                detail = None if image_detail == "omit" else image_detail
                message_content = [
                    {"type": "text", "text": prompt},
                    _image_url_part(image_data_url, detail=detail),
                ]
            else:
                message_content = prompt

            data = {
                "model": model,
                "messages": [
                    {"role": "user", "content": message_content}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }

            response = requests.post(endpoint, headers=headers, json=data, timeout=30)
            response.raise_for_status()

            result = response.json()
            return (_extract_choice_content(result),)

        except requests.exceptions.RequestException as e:
            return (f"Request Error: {str(e)}",)
        except json.JSONDecodeError as e:
            return (f"JSON Error: {str(e)}",)
        except Exception as e:
            return (f"Error: {str(e)}",)


class LLMImageSelectorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe how the candidates should be scored..."
                }),
                "endpoint": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:8080/v1/chat/completions",
                    "placeholder": "OpenAI-compatible chat/completions endpoint"
                }),
                "api_token": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Leave empty for local servers"
                }),
                "model": ("STRING", {
                    "multiline": False,
                    "default": "local-model",
                    "placeholder": "Model name"
                }),
                "candidate_directory": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "/path/to/candidate/images"
                }),
                "recursive_directory": ("BOOLEAN", {
                    "default": False
                }),
                "max_images_per_call": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 32,
                    "step": 1
                }),
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 8192,
                    "step": 1
                }),
                "temperature": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "timeout": ("INT", {
                    "default": 120,
                    "min": 5,
                    "max": 600,
                    "step": 1
                }),
                "grid_columns": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 8,
                    "step": 1
                }),
                "add_id_labels": ("BOOLEAN", {
                    "default": True
                }),
                "return_descriptions": ("BOOLEAN", {
                    "default": False
                }),
                "max_candidate_images": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "candidate_subdirectories": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "llm, auto, or comma-separated folder names"
                }),
                "subdirectory_selection_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Extra instructions for LLM folder selection"
                }),
                "reranker_endpoint": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "http://127.0.0.1:8012"
                }),
                "reranker_model": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Optional reranker model name"
                }),
                "reranker_subdirectory_count": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "candidate_images": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "reference_video": ("IMAGE",),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_SELECTOR_SYSTEM_PROMPT,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("best_image", "candidate_images", "best_index", "best_score", "scores_json", "raw_response")
    FUNCTION = "select_image"
    CATEGORY = "LLM/Image Selection"

    def _headers(self, api_token):
        headers = {"Content-Type": "application/json"}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        return headers

    def _extract_first_json_object(self, text):
        decoder = json.JSONDecoder()
        text = text.strip()

        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[index:])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        raise ValueError("No valid JSON object found in model response")

    def _sample_reference_frames(self, reference_video, max_frames=6):
        frames = _image_batch_to_pil_images(reference_video)
        if len(frames) <= max_frames:
            return frames

        indexes = np.linspace(0, len(frames) - 1, max_frames)
        unique_indexes = []
        for value in indexes:
            index = int(round(float(value)))
            if index not in unique_indexes:
                unique_indexes.append(index)

        return [frames[index] for index in unique_indexes[:max_frames]]

    def _build_prompt_text(self, user_prompt, first_id, last_id, has_reference, add_id_labels):
        reference_text = (
            "Reference image/video parts are attached before the candidate contact sheet."
            if has_reference
            else "No reference image or video was provided; score candidates using the user prompt only."
        )
        id_text = (
            "The candidate contact sheet uses visible 1-based global IDs."
            if add_id_labels
            else "Candidate IDs are assigned by reading order in the contact sheet, left to right, top to bottom."
        )

        return (
            f"{user_prompt.strip()}\n\n"
            f"{reference_text}\n"
            f"Evaluate only candidate IDs {first_id} through {last_id}. "
            f"{id_text}\n\n"
            "Return only valid JSON with this exact schema:\n"
            "{\n"
            '  "candidates": [\n'
            '    {"id": 1, "score": 0-100, "reason": "short reason"},\n'
            '    {"id": 2, "score": 0-100, "reason": "short reason"}\n'
            "  ],\n"
            '  "best_id": 1\n'
            "}\n\n"
            'Rules: "id" must be the visible global candidate ID, "score" must be '
            'a number from 0 to 100, and "best_id" must be one of the candidate IDs '
            "in this request. Do not include markdown fences, comments, or extra text."
        )

    def _post_chunk(
        self,
        endpoint,
        headers,
        model,
        system_prompt,
        content,
        max_tokens,
        temperature,
        timeout,
    ):
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        return _extract_choice_content(response.json())

    def _reranker_urls(self, reranker_endpoint):
        endpoint = str(reranker_endpoint or "").strip()
        if not endpoint:
            return []
        if "://" not in endpoint:
            endpoint = "http://" + endpoint

        endpoint = endpoint.rstrip("/")
        last_segment = endpoint.rsplit("/", 1)[-1]
        if last_segment in {"rerank", "reranking"}:
            return [endpoint]

        return [
            endpoint + "/v1/rerank",
            endpoint + "/v2/rerank",
            endpoint + "/rerank",
            endpoint + "/reranking",
            endpoint + "/v1/reranking",
        ]

    def _reranker_base_url(self, reranker_endpoint):
        endpoint = str(reranker_endpoint or "").strip()
        if not endpoint:
            return ""
        if "://" not in endpoint:
            endpoint = "http://" + endpoint

        endpoint = endpoint.rstrip("/")
        for suffix in ("/v1/reranking", "/v2/rerank", "/v1/rerank", "/reranking", "/rerank"):
            if endpoint.endswith(suffix):
                return endpoint[: -len(suffix)]
        return endpoint

    def _detect_reranker_server(self, reranker_endpoint, headers, timeout):
        base_url = self._reranker_base_url(reranker_endpoint)
        info = {
            "base_url": base_url,
            "version": "",
            "health": "",
            "models": [],
        }
        if not base_url:
            return info

        try:
            response = requests.get(
                base_url + "/version",
                headers=headers,
                timeout=min(max(int(timeout), 1), 10),
            )
            response.raise_for_status()
            data = response.json()
            info["version"] = data.get("version", str(data))
        except Exception:
            pass

        try:
            response = requests.get(
                base_url + "/health",
                headers=headers,
                timeout=min(max(int(timeout), 1), 10),
            )
            if response.ok:
                info["health"] = "ok"
        except Exception:
            pass

        return info

    def _detect_reranker_model(self, reranker_endpoint, headers, reranker_model, timeout):
        reranker_model = str(reranker_model or "").strip()
        if reranker_model:
            return reranker_model, "configured", []

        base_url = self._reranker_base_url(reranker_endpoint)
        if not base_url:
            return "", "none", []

        try:
            response = requests.get(
                base_url + "/v1/models",
                headers=headers,
                timeout=min(max(int(timeout), 1), 10),
            )
            response.raise_for_status()
            models = response.json().get("data", [])
            if models:
                model_id = str(models[0].get("id", "")).strip()
                if model_id:
                    return model_id, "v1/models", models
        except Exception:
            pass

        return "", "none", []

    def _reranker_payload(self, query, documents, reranker_model="", top_n=0):
        payload = {
            "query": query,
            "documents": documents,
        }
        if reranker_model:
            payload["model"] = reranker_model
        if int(top_n) > 0:
            payload["top_n"] = int(top_n)
        payload["return_documents"] = False
        return payload

    def _reranker_query(self, prompt, subdirectory_selection_prompt):
        query = "\n".join(
            part
            for part in [
                str(prompt or "").strip(),
                str(subdirectory_selection_prompt or "").strip(),
            ]
            if part
        )
        return query or "Select the most relevant image candidates for the requested scene."

    def _parse_reranker_response(self, result, document_count):
        if isinstance(result, list):
            raw_results = result
        elif isinstance(result, dict):
            raw_results = result.get("results")
            if raw_results is None:
                raw_results = result.get("data")
            if raw_results is None:
                raw_results = result.get("scores")
        else:
            raw_results = None

        scores = []
        if isinstance(raw_results, list):
            for index, item in enumerate(raw_results):
                if isinstance(item, dict):
                    item_index_value = item.get("index", index)
                    score_value = item.get("relevance_score", item.get("score"))
                else:
                    item_index_value = index
                    score_value = item

                try:
                    item_index = int(item_index_value)
                    if 0 <= item_index < document_count:
                        scores.append((item_index, float(score_value)))
                except (TypeError, ValueError):
                    continue

        return sorted(scores, key=lambda item: item[1], reverse=True)

    def _probe_reranker(self, reranker_endpoint, headers, reranker_model, timeout):
        urls = self._reranker_urls(reranker_endpoint)
        server_info = self._detect_reranker_server(reranker_endpoint, headers, timeout)
        resolved_model, model_source, models = self._detect_reranker_model(
            reranker_endpoint,
            headers,
            reranker_model,
            timeout,
        )
        server_info["models"] = models
        info = {
            "enabled": bool(urls),
            "available": False,
            "url": "",
            "model": resolved_model,
            "model_source": model_source,
            "server": server_info,
            "provider": "unknown",
            "error": "",
        }
        if not urls:
            return None, resolved_model, info

        payload = self._reranker_payload(
            query="test",
            documents=["test"],
            reranker_model=resolved_model,
            top_n=1,
        )
        for url in urls:
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=min(max(int(timeout), 1), 10),
                )
                response.raise_for_status()
                scores = self._parse_reranker_response(response.json(), 1)
                if scores:
                    info["available"] = True
                    info["url"] = url
                    if url.endswith("/v2/rerank"):
                        info["provider"] = "vllm_v2_or_cohere_style"
                    elif url.endswith("/v1/rerank") and isinstance(response.json(), list):
                        info["provider"] = "sglang_or_vllm_style"
                    elif url.endswith("/v1/rerank"):
                        info["provider"] = "vllm_llamacpp_or_openai_style"
                    elif "reranking" in url or url.endswith("/rerank"):
                        info["provider"] = "vllm_llamacpp_style"
                    return url, resolved_model, info
                info["error"] = f"{url}: response did not contain rerank scores"
            except Exception as exc:
                info["error"] = f"{url}: {exc}"

        return None, resolved_model, info

    def _rerank_documents(self, reranker_url, headers, reranker_model, query, documents, top_n, timeout):
        response = requests.post(
            reranker_url,
            headers=headers,
            json=self._reranker_payload(
                query=query,
                documents=documents,
                reranker_model=reranker_model,
                top_n=top_n,
            ),
            timeout=timeout,
        )
        response.raise_for_status()
        scores = self._parse_reranker_response(response.json(), len(documents))
        if not scores:
            raise ValueError("Reranker response did not contain usable scores")
        return scores

    def _list_candidate_subdirectories(self, candidate_directory):
        directory = str(candidate_directory or "").strip()
        if not directory:
            return []

        root = Path(directory).expanduser()
        if not root.exists():
            raise ValueError(f"candidate_directory does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"candidate_directory is not a directory: {root}")

        return [
            path.relative_to(root).as_posix()
            for path in sorted(root.rglob("*"))
            if path.is_dir()
        ]

    def _match_llm_selected_subdirectories(self, selected_names, available_subdirectories):
        matched = []
        for selected_name in selected_names:
            selected_text = _normalize_filter_text(selected_name)
            if not selected_text:
                continue

            for available in available_subdirectories:
                available_text = _normalize_filter_text(available)
                available_name_text = _normalize_filter_text(Path(available).name)
                if (
                    selected_text == available_text
                    or selected_text == available_name_text
                    or selected_text in available_text
                    or selected_text in available_name_text
                    or available_name_text in selected_text
                ):
                    if available not in matched:
                        matched.append(available)

        return matched

    def _select_candidate_subdirectories_with_llm(
        self,
        prompt,
        candidate_directory,
        endpoint,
        headers,
        model,
        reference_parts,
        subdirectory_selection_prompt,
        max_tokens,
        timeout,
    ):
        available_subdirectories = self._list_candidate_subdirectories(candidate_directory)
        selection_info = {
            "mode": "llm",
            "available_subdirectories": available_subdirectories,
            "selected_subdirectories": [],
            "raw_response": "",
            "fallback_to_all": False,
        }
        if not available_subdirectories:
            selection_info["fallback_to_all"] = True
            selection_info["error"] = "No subdirectories found"
            return "", selection_info

        folder_list = "\n".join(f"- {name}" for name in available_subdirectories)
        content = [
            {
                "type": "text",
                "text": (
                    "Choose the candidate subfolders that are relevant for this image "
                    "selection request. You may choose one or multiple folders. If no "
                    "specific folder is clearly relevant, return an empty list so all "
                    "folders can be searched.\n\n"
                    f"User prompt:\n{prompt.strip()}\n\n"
                    f"Extra folder/style instructions:\n{subdirectory_selection_prompt.strip()}\n\n"
                    f"Available subfolders:\n{folder_list}\n\n"
                    "Return only valid JSON with this exact schema:\n"
                    '{ "subdirectories": ["relative/folder/name"] }'
                ),
            },
        ]
        content.extend(reference_parts)

        try:
            raw_text = self._post_chunk(
                endpoint=endpoint,
                headers=headers,
                model=model,
                system_prompt=SUBDIRECTORY_SELECTOR_SYSTEM_PROMPT,
                content=content,
                max_tokens=max(64, min(int(max_tokens), 1024)),
                temperature=0.0,
                timeout=timeout,
            )
            selection_info["raw_response"] = raw_text
            parsed = self._extract_first_json_object(raw_text)
            selected_names = parsed.get("subdirectories", [])
            if not isinstance(selected_names, list):
                raise ValueError('JSON response must contain a "subdirectories" list')

            matched = self._match_llm_selected_subdirectories(
                selected_names,
                available_subdirectories,
            )
            selection_info["selected_subdirectories"] = matched
            if not matched:
                selection_info["fallback_to_all"] = True
                return "", selection_info

            return ",".join(matched), selection_info
        except Exception as exc:
            selection_info["fallback_to_all"] = True
            selection_info["error"] = str(exc)
            return "", selection_info

    def _select_candidate_subdirectories_with_reranker(
        self,
        prompt,
        candidate_directory,
        reranker_url,
        headers,
        reranker_model,
        subdirectory_selection_prompt,
        reranker_subdirectory_count,
        timeout,
    ):
        available_subdirectories = self._list_candidate_subdirectories(candidate_directory)
        selection_info = {
            "mode": "reranker",
            "available_subdirectories": available_subdirectories,
            "selected_subdirectories": [],
            "scores": [],
            "fallback_to_all": False,
        }
        if not available_subdirectories:
            selection_info["fallback_to_all"] = True
            selection_info["error"] = "No subdirectories found"
            return "", selection_info

        query = self._reranker_query(prompt, subdirectory_selection_prompt)
        documents = [
            f"Candidate image folder: {name}"
            for name in available_subdirectories
        ]

        try:
            scores = self._rerank_documents(
                reranker_url=reranker_url,
                headers=headers,
                reranker_model=reranker_model,
                query=query,
                documents=documents,
                top_n=reranker_subdirectory_count,
                timeout=timeout,
            )
            selected = [
                available_subdirectories[index]
                for index, _score in scores[:int(reranker_subdirectory_count)]
            ]
            selection_info["selected_subdirectories"] = selected
            selection_info["scores"] = [
                {
                    "subdirectory": available_subdirectories[index],
                    "score": score,
                }
                for index, score in scores
            ]
            if not selected:
                selection_info["fallback_to_all"] = True
                return "", selection_info

            return ",".join(selected), selection_info
        except Exception as exc:
            selection_info["fallback_to_all"] = True
            selection_info["error"] = str(exc)
            return "", selection_info

    def _collect_candidate_images(
        self,
        prompt,
        candidate_directory,
        recursive_directory,
        candidate_subdirectories,
        candidate_images,
        image=None,
    ):
        candidate_pil_images = []
        candidate_sources = []

        if image is not None:
            input_images = _image_batch_to_pil_images(image)
            for batch_index, input_image in enumerate(input_images):
                candidate_sources.append({
                    "type": "image_input",
                    "batch_index": batch_index,
                })
                candidate_pil_images.append(input_image)
            return candidate_pil_images, candidate_sources

        directory_images, directory_paths = _load_images_from_directory(
            candidate_directory,
            recursive=recursive_directory,
            subdirectory_filter=candidate_subdirectories,
            prompt=prompt,
        )
        for image, path in zip(directory_images, directory_paths):
            candidate_sources.append({
                "type": "directory",
                "path": path,
            })
            candidate_pil_images.append(image)

        if candidate_images is not None:
            batch_images = _image_batch_to_pil_images(candidate_images)
            for batch_index, image in enumerate(batch_images):
                candidate_sources.append({
                    "type": "input_batch",
                    "batch_index": batch_index,
                })
                candidate_pil_images.append(image)

        return candidate_pil_images, candidate_sources

    def _limit_candidate_images(self, candidate_pil_images, candidate_sources, max_candidate_images, seed=None):
        max_candidate_images = int(max_candidate_images)
        if max_candidate_images <= 0 or len(candidate_pil_images) <= max_candidate_images:
            return candidate_pil_images, candidate_sources

        r = random.Random(seed) if seed is not None else random
        sampled_indexes = r.sample(range(len(candidate_pil_images)), max_candidate_images)
        return (
            [candidate_pil_images[index] for index in sampled_indexes],
            [candidate_sources[index] for index in sampled_indexes],
        )

    def _candidate_source_document(self, source):
        if source.get("type") == "directory":
            path = Path(source.get("path", ""))
            return (
                f"Candidate image file: {path.name}\n"
                f"Folder: {path.parent.as_posix()}\n"
                f"Name without extension: {path.stem}"
            )

        if source.get("type") == "input_batch":
            return f"Candidate image from input batch index {source.get('batch_index', 0)}"

        if source.get("type") == "image_input":
            return f"Direct candidate image input index {source.get('batch_index', 0)}"

        return json.dumps(source, ensure_ascii=False)

    def _rerank_candidate_images(
        self,
        candidate_pil_images,
        candidate_sources,
        reranker_url,
        headers,
        reranker_model,
        prompt,
        subdirectory_selection_prompt,
        max_candidate_images,
        timeout,
    ):
        info = {
            "used": False,
            "top_n": int(max_candidate_images),
            "scores": [],
            "error": "",
        }
        if not reranker_url or not candidate_pil_images:
            return candidate_pil_images, candidate_sources, info

        query = self._reranker_query(prompt, subdirectory_selection_prompt)
        documents = [
            self._candidate_source_document(source)
            for source in candidate_sources
        ]
        top_n = int(max_candidate_images) if int(max_candidate_images) > 0 else 0

        try:
            scores = self._rerank_documents(
                reranker_url=reranker_url,
                headers=headers,
                reranker_model=reranker_model,
                query=query,
                documents=documents,
                top_n=top_n,
                timeout=timeout,
            )
            selected_scores = scores[:top_n] if top_n > 0 else scores
            selected_indexes = [index for index, _score in selected_scores]
            info["used"] = True
            info["scores"] = [
                {
                    "index": index,
                    "score": score,
                    "source": candidate_sources[index],
                }
                for index, score in scores
            ]
            return (
                [candidate_pil_images[index] for index in selected_indexes],
                [candidate_sources[index] for index in selected_indexes],
                info,
            )
        except Exception as exc:
            info["error"] = str(exc)
            return candidate_pil_images, candidate_sources, info

    def _select_original_candidate(self, image, candidate_images, candidate_pil_images, candidate_sources, zero_index):
        source = candidate_sources[zero_index]
        if source["type"] == "image_input" and image is not None:
            batch_index = source["batch_index"]
            if len(image.shape) == 4:
                return image[batch_index:batch_index + 1]
            if hasattr(image, "unsqueeze"):
                return image.unsqueeze(0)
            return np.expand_dims(image, axis=0)

        if source["type"] == "input_batch" and candidate_images is not None:
            batch_index = source["batch_index"]
            if len(candidate_images.shape) == 4:
                return candidate_images[batch_index:batch_index + 1]
            if hasattr(candidate_images, "unsqueeze"):
                return candidate_images.unsqueeze(0)
            return np.expand_dims(candidate_images, axis=0)

        return _pil_image_to_comfy_image(candidate_pil_images[zero_index])

    def select_image(
        self,
        prompt,
        endpoint,
        api_token,
        model,
        candidate_directory,
        recursive_directory,
        max_images_per_call,
        max_tokens,
        temperature,
        timeout,
        grid_columns,
        add_id_labels,
        return_descriptions,
        max_candidate_images,
        seed,
        candidate_subdirectories,
        subdirectory_selection_prompt,
        reranker_endpoint,
        reranker_model,
        reranker_subdirectory_count,
        image=None,
        candidate_images=None,
        reference_image=None,
        reference_video=None,
        system_prompt=DEFAULT_SELECTOR_SYSTEM_PROMPT,
    ):
        headers = self._headers(api_token)
        reranker_url, resolved_reranker_model, reranker_info = self._probe_reranker(
            reranker_endpoint=reranker_endpoint,
            headers=headers,
            reranker_model=reranker_model,
            timeout=timeout,
        )
        max_images_per_call = max(1, min(int(max_images_per_call), 32))
        grid_columns = max(1, min(int(grid_columns), 8))

        reference_parts = []
        if reference_image is not None:
            reference_images = _image_batch_to_pil_images(reference_image)
            reference_parts.append(_image_url_part(_encode_pil_to_data_url(reference_images[0])))

        if reference_video is not None:
            reference_frames = self._sample_reference_frames(reference_video, max_frames=6)
            reference_labels = [f"REF-{index + 1}" for index in range(len(reference_frames))]
            reference_sheet = _build_contact_sheet(reference_frames, grid_columns, reference_labels)
            reference_parts.append(_image_url_part(_encode_pil_to_data_url(reference_sheet)))

        candidate_subdirectory_selection = None
        resolved_candidate_subdirectories = candidate_subdirectories
        candidate_subdirectories_mode = str(candidate_subdirectories or "").strip().casefold()
        if image is None and candidate_subdirectories_mode == "reranker":
            if reranker_url:
                resolved_candidate_subdirectories, candidate_subdirectory_selection = (
                    self._select_candidate_subdirectories_with_reranker(
                        prompt=prompt,
                        candidate_directory=candidate_directory,
                        reranker_url=reranker_url,
                        headers=headers,
                        reranker_model=resolved_reranker_model,
                        subdirectory_selection_prompt=subdirectory_selection_prompt,
                        reranker_subdirectory_count=reranker_subdirectory_count,
                        timeout=timeout,
                    )
                )
            else:
                candidate_subdirectory_selection = {
                    "mode": "reranker",
                    "fallback_to_all": True,
                    "error": reranker_info.get("error", "Reranker endpoint is not available"),
                }
                resolved_candidate_subdirectories = ""
        elif image is None and candidate_subdirectories_mode == "llm":
            resolved_candidate_subdirectories, candidate_subdirectory_selection = (
                self._select_candidate_subdirectories_with_llm(
                    prompt=prompt,
                    candidate_directory=candidate_directory,
                    endpoint=endpoint,
                    headers=headers,
                    model=model,
                    reference_parts=reference_parts,
                    subdirectory_selection_prompt=subdirectory_selection_prompt,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            )

        try:
            candidate_pil_images, candidate_sources = self._collect_candidate_images(
                prompt=prompt,
                candidate_directory=candidate_directory,
                recursive_directory=recursive_directory,
                candidate_subdirectories=resolved_candidate_subdirectories,
                candidate_images=candidate_images,
                image=image,
            )
        except ValueError as exc:
            if candidate_subdirectory_selection is None:
                raise

            candidate_subdirectory_selection["fallback_to_all"] = True
            candidate_subdirectory_selection["fallback_error"] = str(exc)
            resolved_candidate_subdirectories = ""
            candidate_pil_images, candidate_sources = self._collect_candidate_images(
                prompt=prompt,
                candidate_directory=candidate_directory,
                recursive_directory=recursive_directory,
                candidate_subdirectories=resolved_candidate_subdirectories,
                candidate_images=candidate_images,
                image=image,
            )
        if not candidate_pil_images:
            raise ValueError(
                "No candidate images found. Connect image, candidate_images, set candidate_directory, or use both."
            )

        original_candidate_count = len(candidate_pil_images)
        reranker_candidate_selection = None
        if image is None:
            if reranker_url:
                (
                    candidate_pil_images,
                    candidate_sources,
                    reranker_candidate_selection,
                ) = self._rerank_candidate_images(
                    candidate_pil_images=candidate_pil_images,
                    candidate_sources=candidate_sources,
                    reranker_url=reranker_url,
                    headers=headers,
                    reranker_model=resolved_reranker_model,
                    prompt=prompt,
                    subdirectory_selection_prompt=subdirectory_selection_prompt,
                    max_candidate_images=max_candidate_images,
                    timeout=timeout,
                )

            if not reranker_candidate_selection or not reranker_candidate_selection.get("used"):
                candidate_pil_images, candidate_sources = self._limit_candidate_images(
                    candidate_pil_images,
                    candidate_sources,
                    max_candidate_images,
                    seed=seed,
                )

        # Convert the actually used candidate images to a ComfyUI batch
        candidate_batch = _pil_list_to_comfy_batch(candidate_pil_images)

        scores_by_id = {}
        raw_responses = []
        failures = []
        candidate_count = len(candidate_pil_images)

        for start in range(0, candidate_count, max_images_per_call):
            end = min(start + max_images_per_call, candidate_count)
            chunk_images = candidate_pil_images[start:end]
            global_ids = list(range(start + 1, end + 1))
            labels = global_ids if add_id_labels else None
            candidate_sheet = _build_contact_sheet(chunk_images, grid_columns, labels)

            content = [
                {
                    "type": "text",
                    "text": self._build_prompt_text(
                        prompt,
                        first_id=global_ids[0],
                        last_id=global_ids[-1],
                        has_reference=bool(reference_parts),
                        add_id_labels=add_id_labels,
                    ),
                },
            ]
            content.extend(reference_parts)
            content.append(_image_url_part(_encode_pil_to_data_url(candidate_sheet)))

            chunk_info = {
                "candidate_ids": global_ids,
                "raw_response": "",
            }

            try:
                raw_text = self._post_chunk(
                    endpoint=endpoint,
                    headers=headers,
                    model=model,
                    system_prompt=system_prompt,
                    content=content,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                )
                chunk_info["raw_response"] = raw_text
                parsed = self._extract_first_json_object(raw_text)
                parsed_candidates = parsed.get("candidates")
                if not isinstance(parsed_candidates, list) or not parsed_candidates:
                    raise ValueError('JSON response must contain a non-empty "candidates" list')

                chunk_ids = set(global_ids)
                for item in parsed_candidates:
                    if not isinstance(item, dict):
                        continue

                    try:
                        candidate_id = int(item.get("id"))
                        score = float(item.get("score"))
                    except (TypeError, ValueError):
                        continue

                    if candidate_id not in chunk_ids:
                        continue

                    score = max(0.0, min(100.0, score))
                    reason = str(item.get("reason", "")).strip()

                    scores_by_id[candidate_id] = {
                        "one_based_id": candidate_id,
                        "zero_based_index": candidate_id - 1,
                        "score": score,
                        "reason": reason if return_descriptions else "",
                        "source": candidate_sources[candidate_id - 1],
                    }

                if not any(candidate_id in scores_by_id for candidate_id in global_ids):
                    raise ValueError("No usable candidate scores found for this chunk")

            except Exception as exc:
                chunk_info["error"] = str(exc)
                failures.append({
                    "candidate_ids": global_ids,
                    "error": str(exc),
                    "raw_response": chunk_info.get("raw_response", ""),
                })
            finally:
                raw_responses.append(chunk_info)

        if not scores_by_id:
            failure_text = json.dumps(failures, ensure_ascii=False, indent=2)
            raise RuntimeError(f"All LLM image selection chunks failed: {failure_text}")

        best_record = max(
            scores_by_id.values(),
            key=lambda item: (item["score"], -item["zero_based_index"]),
        )
        best_index = int(best_record["zero_based_index"])
        best_score = float(best_record["score"])
        best_image = self._select_original_candidate(
            image,
            candidate_images,
            candidate_pil_images,
            candidate_sources,
            best_index,
        )

        scores_payload = {
            "best_index": best_index,
            "best_id": best_index + 1,
            "best_score": best_score,
            "best_source": candidate_sources[best_index],
            "candidate_count": candidate_count,
            "original_candidate_count": original_candidate_count,
            "max_candidate_images": int(max_candidate_images),
            "seed": int(seed),
            "candidate_directory": str(candidate_directory or "").strip(),
            "candidate_subdirectories": str(candidate_subdirectories or "").strip(),
            "subdirectory_selection_prompt": str(subdirectory_selection_prompt or "").strip(),
            "resolved_candidate_subdirectories": str(resolved_candidate_subdirectories or "").strip(),
            "candidate_subdirectory_selection": candidate_subdirectory_selection,
            "reranker": reranker_info,
            "reranker_candidate_selection": reranker_candidate_selection,
            "recursive_directory": bool(recursive_directory),
            "scored_count": len(scores_by_id),
            "candidates": [
                scores_by_id[candidate_id]
                for candidate_id in sorted(scores_by_id.keys())
            ],
            "failures": failures,
        }

        return (
            best_image,
            candidate_batch,
            best_index,
            best_score,
            json.dumps(scores_payload, ensure_ascii=False, indent=2),
            json.dumps(raw_responses, ensure_ascii=False, indent=2),
        )
