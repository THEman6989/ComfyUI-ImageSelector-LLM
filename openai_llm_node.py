import base64
import io
import json
import math
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
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    try:
        import torch

        return torch.from_numpy(image_array)
    except Exception:
        return image_array


def _load_images_from_directory(directory, recursive=False):
    """Load every supported image file from a directory as RGB PIL images."""
    directory = str(directory or "").strip()
    if not directory:
        return [], []

    root = Path(directory).expanduser()
    if not root.exists():
        raise ValueError(f"candidate_directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"candidate_directory is not a directory: {root}")

    iterator = root.rglob("*") if recursive else root.iterdir()
    image_paths = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
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
            },
            "optional": {
                "candidate_images": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "reference_video": ("IMAGE",),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_SELECTOR_SYSTEM_PROMPT,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("best_image", "best_index", "best_score", "scores_json", "raw_response")
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

    def _collect_candidate_images(self, candidate_directory, recursive_directory, candidate_images):
        candidate_pil_images = []
        candidate_sources = []

        directory_images, directory_paths = _load_images_from_directory(
            candidate_directory,
            recursive=recursive_directory,
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

    def _select_original_candidate(self, candidate_images, candidate_pil_images, candidate_sources, zero_index):
        source = candidate_sources[zero_index]
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
        candidate_images=None,
        reference_image=None,
        reference_video=None,
        system_prompt=DEFAULT_SELECTOR_SYSTEM_PROMPT,
    ):
        candidate_pil_images, candidate_sources = self._collect_candidate_images(
            candidate_directory=candidate_directory,
            recursive_directory=recursive_directory,
            candidate_images=candidate_images,
        )
        if not candidate_pil_images:
            raise ValueError(
                "No candidate images found. Connect candidate_images, set candidate_directory, or use both."
            )

        headers = self._headers(api_token)
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
            "candidate_directory": str(candidate_directory or "").strip(),
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
            best_index,
            best_score,
            json.dumps(scores_payload, ensure_ascii=False, indent=2),
            json.dumps(raw_responses, ensure_ascii=False, indent=2),
        )
