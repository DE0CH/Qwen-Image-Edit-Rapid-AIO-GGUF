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

DEFAULT_MODEL_URL = (
    "https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO"
    "/blob/main/v23/Qwen-Rapid-AIO-SFW-v23.safetensors"
)
DEFAULT_BASE_REPO = "Qwen/Qwen-Image-Edit-2511"

MAX_IMAGES_IN = 3
MAX_IMAGES_OUT = 4
MAX_STEPS = 50


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

    # FP8 all-in-one safetensors: load the transformer keys keeping the
    # checkpoint's FP8 storage dtype, and upcast per-layer at compute time.
    transformer = QwenImageTransformer2DModel.from_single_file(
        model_url,
        config=base_repo,
        subfolder="transformer",
    )
    transformer.enable_layerwise_casting(
        storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
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
