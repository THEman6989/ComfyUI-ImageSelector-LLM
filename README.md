# ComfyUI LLM Image Selector

ComfyUI custom nodes for OpenAI-compatible chat/completions endpoints.

The main node in this fork is **LLM Image Selector**. It sends labelled contact sheets of candidate images to a vision-capable LLM and asks the model to score which candidate best matches a prompt, reference image, or sampled reference video frames.

The original **OpenAI Compatible LLM** node is still registered for backward compatibility.

## Installation

Clone this repository into `ComfyUI/custom_nodes`:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/THEman6989/ComfyUI-ImageSelector-LLM.git
cd ComfyUI-ImageSelector-LLM
pip install -r requirements.txt
```

Restart ComfyUI after installation.

## Nodes

### LLM Image Selector

Category: `LLM/Image Selection`

Inputs:

| Input | Type | Description |
| --- | --- | --- |
| `prompt` | STRING | Multiline instructions for how candidates should be judged. |
| `endpoint` | STRING | OpenAI-compatible `/v1/chat/completions` endpoint. |
| `api_token` | STRING | Bearer token. Leave empty for local servers that do not require auth. |
| `model` | STRING | Model name sent in the request body. |
| `candidate_directory` | STRING | Optional folder path. Every supported image file in the folder becomes a candidate. |
| `recursive_directory` | BOOLEAN | Also load images from subfolders when `candidate_directory` is set. |
| `image` | IMAGE | Optional direct image input. When connected, this image is sent to the API instead of random/directory/batch candidates. |
| `candidate_images` | IMAGE | Optional ComfyUI image batch `[B,H,W,C]`; each batch item is one candidate. |
| `max_images_per_call` | INT | Number of candidates per LLM request. Default `8`. |
| `max_tokens` | INT | Maximum response tokens. Default `1024`. |
| `temperature` | FLOAT | Sampling temperature. Default `0.0` for stable scoring. |
| `timeout` | INT | Request timeout in seconds. Default `120`. |
| `grid_columns` | INT | Contact sheet column count. Default `4`. |
| `add_id_labels` | BOOLEAN | Draw visible 1-based candidate IDs on the contact sheet. |
| `return_descriptions` | BOOLEAN | Include model reasons in `scores_json`. |
| `max_candidate_images` | INT | Randomly limit the candidate pool before scoring. `0` means no limit. |
| `reference_image` | IMAGE | Optional reference image attached to every request. |
| `reference_video` | IMAGE | Optional IMAGE batch treated as video frames; up to 6 frames are sampled. |
| `system_prompt` | STRING | Optional judge/system instructions. |

Outputs:

| Output | Type | Meaning |
| --- | --- | --- |
| `best_image` | IMAGE | The original unlabelled candidate image selected by the LLM. |
| `best_index` | INT | Zero-based index of the selected candidate. |
| `best_score` | FLOAT | Best score from `0` to `100`. |
| `scores_json` | STRING | Structured scores, including `zero_based_index` and `one_based_id`. |
| `raw_response` | STRING | Raw model responses per chunk for debugging. |

### OpenAI Compatible LLM

Category: `LLM`

This is the original text/image prompt node. It remains available as **OpenAI Compatible LLM**. If a batch of images is connected to this older node, the batch is now encoded as a contact sheet instead of silently using only the first image.

## llama.cpp Endpoint Example

Start a llama.cpp server with a vision-capable model and projector, then point the node at the local OpenAI-compatible endpoint:

```bash
llama-server \
  -m /path/to/model.gguf \
  --mmproj /path/to/mmproj.gguf \
  --host 127.0.0.1 \
  --port 8080
```

Node settings:

```text
endpoint: http://127.0.0.1:8080/v1/chat/completions
api_token:
model: local-model
temperature: 0.0
```

Leave `api_token` empty unless your server requires authentication.

## Visual Scoring, Not Tool Calling

This node does not use real OpenAI tool/function calling. It uses normal multimodal chat content: text plus base64 PNG `image_url` parts. The model is instructed to return strict JSON with candidate scores. The node parses that JSON and routes the selected image through the `best_image` output.

Because the final choice is model-generated visual scoring, quality depends on the vision model, prompt clarity, and image layout.

## Example Prompt For Outfit Matching

```text
Choose the candidate whose outfit best matches the reference.
Focus on jacket shape, shirt color, pants/skirt color, shoes, accessories,
patterns, and overall silhouette. Ignore pose, camera angle, background,
lighting, facial expression, and image quality.
```

## Chunking With max_images_per_call

Candidates can come from `image`, `candidate_directory`, `candidate_images`, or a combination of directory and batch candidates. When the direct `image` input is connected, it takes priority and is used as the candidate source instead of directory or `candidate_images` inputs. Without `image`, directory files are loaded first, sorted by path, then any connected ComfyUI IMAGE batch is appended. Supported file extensions are `bmp`, `gif`, `jpg`, `jpeg`, `png`, `tif`, `tiff`, and `webp`.

`candidate_images` is a ComfyUI IMAGE batch shaped `[B,H,W,C]`. The selector preserves every candidate in the batch, adds optional visible labels `1`, `2`, `3`, and builds contact sheet grids.

Set `max_candidate_images` to randomly sample a smaller pool before any LLM calls are made. For example, if a folder contains 1000 images and `max_candidate_images` is `30`, the node scores only 30 randomly selected candidates. If the pool has fewer images than the limit, all candidates are used. `0` disables the limit.

If the candidate set is larger than `max_images_per_call`, the node sends multiple requests internally. For example, 20 candidates and `max_images_per_call=8` produces three calls: candidates `1-8`, `9-16`, and `17-20`. ComfyUI does not need a workflow loop for this.

Scores are merged by global candidate ID. The returned `best_index` is zero-based for programmatic use, while `scores_json` also includes the visible one-based IDs used in the contact sheets. For directory candidates, `scores_json` includes the original file path in each candidate `source`.

## JSON Response Expected From The Model

The prompt asks the model to return only:

```json
{
  "candidates": [
    {"id": 1, "score": 0, "reason": "short reason"},
    {"id": 2, "score": 100, "reason": "short reason"}
  ],
  "best_id": 2
}
```

The parser tolerates markdown fences or extra surrounding text by extracting the first valid JSON object. If one chunk fails, the node records that raw response and continues with other chunks when possible. If every chunk fails, it raises a clear exception.

## Requirements

- ComfyUI
- Python 3.8+
- requests
- Pillow
- numpy

No heavy extra dependencies are required; ComfyUI already provides the tensor objects used for IMAGE values.
