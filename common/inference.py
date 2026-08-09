"""GPU inference for Qwen-Image-Edit Rapid AIO (GGUF transformer + diffusers pipeline).

The GGUF file from Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF contains only the
diffusion transformer (with the Lightning/Rapid merges baked in, so 4 steps and
CFG 1.0 are enough). The text encoder, VAE and processor come from the base
Qwen/Qwen-Image-Edit-2511 repository.

Configuration via environment variables:
  GGUF_URL   - Hugging Face blob URL of the .gguf transformer to load
  BASE_REPO  - base pipeline repo (default Qwen/Qwen-Image-Edit-2511)
  LOW_VRAM   - set to any value to enable model CPU offload
"""

import os

DEFAULT_GGUF_URL = (
    "https://huggingface.co/Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF"
    "/blob/main/v90/qwen-rapid-nsfw-v9.0-Q4_K_M.gguf"
)
DEFAULT_BASE_REPO = "Qwen/Qwen-Image-Edit-2511"

MAX_IMAGES_IN = 3
MAX_IMAGES_OUT = 4
MAX_STEPS = 50


def load_pipeline(device: str = "cuda"):
    import torch
    from diffusers import (
        GGUFQuantizationConfig,
        QwenImageEditPlusPipeline,
        QwenImageTransformer2DModel,
    )

    gguf_url = os.environ.get("GGUF_URL", DEFAULT_GGUF_URL)
    base_repo = os.environ.get("BASE_REPO", DEFAULT_BASE_REPO)

    transformer = QwenImageTransformer2DModel.from_single_file(
        gguf_url,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        config=base_repo,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
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
