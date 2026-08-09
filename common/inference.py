"""GPU inference for Qwen-Image-Edit Rapid AIO.

Default model: Phr00t/Qwen-Image-Edit-Rapid-AIO — an FP8 all-in-one
checkpoint (transformer with Lightning/Rapid merges baked in, so 4 steps and
CFG 1.0 are enough). Only the transformer is taken from the AIO file; the
text encoder, VAE and processor come from the base Qwen/Qwen-Image-Edit-2511
repository. FP8 weights are kept in FP8 storage with bfloat16 compute via
diffusers layerwise casting.

GGUF checkpoints (e.g. Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF) are also
supported: point MODEL_URL at a .gguf file.

Configuration via environment variables:
  MODEL_URL  - Hugging Face blob URL of the checkpoint (.safetensors or .gguf)
  GGUF_URL   - legacy alias for MODEL_URL
  BASE_REPO  - base pipeline repo (default Qwen/Qwen-Image-Edit-2511)
  LOW_VRAM   - set to any value to enable model CPU offload
"""

import os
import re

DEFAULT_MODEL_URL = (
    "https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO"
    "/blob/main/v23/Qwen-Rapid-AIO-NSFW-v23.safetensors"
)
DEFAULT_BASE_REPO = "Qwen/Qwen-Image-Edit-2511"

MAX_IMAGES_IN = 3
MAX_IMAGES_OUT = 4
MAX_STEPS = 50


_AIO_PREFIX = "model.diffusion_model."


def _parse_hf_url(url: str):
    """Split an hf.co blob/resolve URL into (repo_id, revision, filename)."""
    m = re.match(
        r"https?://huggingface\.co/([^/]+/[^/]+)/(?:blob|resolve)/([^/]+)/(.+?)(?:\?.*)?$",
        url,
    )
    if not m:
        raise ValueError(f"not a huggingface.co file URL: {url}")
    return m.group(1), m.group(2), m.group(3)


def _load_transformer(model_url: str, base_repo: str):
    import torch
    from diffusers import QwenImageTransformer2DModel

    if model_url.endswith(".gguf"):
        from diffusers import GGUFQuantizationConfig

        return QwenImageTransformer2DModel.from_single_file(
            model_url,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            config=base_repo,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )

    # FP8 all-in-one safetensors (ComfyUI checkpoint layout). The transformer
    # keys are diffusers-style names under a "model.diffusion_model." prefix,
    # so load them directly: strip the prefix, skip marker keys, and assign
    # keeping the checkpoint's FP8 storage dtype. diffusers' from_single_file
    # mis-detects this multi-component file and leaves meta tensors behind.
    from accelerate import init_empty_weights
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    repo_id, revision, filename = _parse_hf_url(model_url)
    path = hf_hub_download(repo_id, filename, revision=revision)

    config = QwenImageTransformer2DModel.load_config(base_repo, subfolder="transformer")
    with init_empty_weights(include_buffers=False):
        transformer = QwenImageTransformer2DModel.from_config(config)

    state = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if not key.startswith(_AIO_PREFIX):
                continue
            name = key[len(_AIO_PREFIX):]
            if name.startswith("__"):  # marker keys like __index_timestep_zero__
                continue
            state[name] = f.get_tensor(key)

    missing, unexpected = transformer.load_state_dict(state, strict=False, assign=True)
    missing = [k for k in missing if not k.endswith("_extra_state")]
    if missing:
        raise RuntimeError(
            f"AIO checkpoint is missing {len(missing)} transformer keys, "
            f"e.g. {missing[:3]}"
        )
    if unexpected:
        print(f"[inference] ignoring {len(unexpected)} unexpected keys: {unexpected[:3]}")

    transformer.enable_layerwise_casting(
        storage_dtype=torch.float8_e4m3fn,
        compute_dtype=torch.bfloat16,
        # the AIO stores *all* weights in FP8, including norm layers that the
        # default pattern would skip (and then crash on Float x Float8 math)
        skip_modules_pattern=(),
    )
    return transformer


def load_pipeline(device: str = "cuda"):
    import torch
    from diffusers import QwenImageEditPlusPipeline

    model_url = (
        os.environ.get("MODEL_URL")
        or os.environ.get("GGUF_URL")
        or DEFAULT_MODEL_URL
    )
    base_repo = os.environ.get("BASE_REPO", DEFAULT_BASE_REPO)

    transformer = _load_transformer(model_url, base_repo)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        base_repo, transformer=transformer, torch_dtype=torch.bfloat16
    )
    if os.environ.get("LOW_VRAM"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    return pipe


def run_edit(
    pipe,
    images,
    prompt: str,
    negative_prompt: str = " ",
    seed: int = 0,
    true_cfg_scale: float = 1.0,
    num_inference_steps: int = 4,
    num_images: int = 1,
):
    import torch

    from common.imgcodec import limit_size

    images = [limit_size(img.convert("RGB")) for img in images[:MAX_IMAGES_IN]]
    num_inference_steps = max(1, min(int(num_inference_steps), MAX_STEPS))
    num_images = max(1, min(int(num_images), MAX_IMAGES_OUT))

    result = pipe(
        image=images,
        prompt=prompt,
        negative_prompt=negative_prompt or " ",
        num_inference_steps=num_inference_steps,
        true_cfg_scale=float(true_cfg_scale),
        num_images_per_prompt=num_images,
        generator=torch.Generator(device="cpu").manual_seed(int(seed)),
    )
    return result.images
