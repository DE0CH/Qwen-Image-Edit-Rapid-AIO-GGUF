"""Modal deployment for Qwen-Image-Edit Rapid AIO (GGUF).

Deploy with:  modal deploy modal_app.py
Warm up with: modal run modal_app.py::warmup

Serves (on one https://<workspace>--qwen-image-edit-rapid-web.modal.run URL):
  /            Gradio web UI
  /v1/edit     JSON API: {prompt, images: [b64 png/jpeg, ...], seed?,
                          negative_prompt?, true_cfg_scale?,
                          num_inference_steps?, num_images?}
               -> {images: [b64 png, ...], seed}
  /health      liveness check

Deploy-time environment variables (all optional, baked in at deploy):
  MODAL_GPU        GPU type for inference (default L40S)
  GGUF_URL         which .gguf file to serve (default v90 Q4_K_M)
  BASE_REPO        base diffusers repo (default Qwen/Qwen-Image-Edit-2511)
  LOW_VRAM         enable CPU offload (for smaller GPUs)
  HF_TOKEN         Hugging Face token for downloads (not usually needed)
  MODAL_AUTH_TOKEN if set, /v1/edit requires "Authorization: Bearer <token>"
                   and the web UI requires login (user: "user", password: token)
"""

import os
import random

import modal

APP_NAME = "qwen-image-edit-rapid"
CACHE_DIR = "/cache"
GPU_TYPE = os.environ.get("MODAL_GPU") or "L40S"

MAX_SEED = 2**31 - 1

hf_cache = modal.Volume.from_name(f"{APP_NAME}-hf-cache", create_if_missing=True)

# Deploy-time configuration forwarded into the containers.
model_env = {
    k: v
    for k in ("GGUF_URL", "BASE_REPO", "LOW_VRAM", "HF_TOKEN")
    if (v := os.environ.get(k))
}
model_secret = modal.Secret.from_dict(model_env)
web_secret = modal.Secret.from_dict(
    {"MODAL_AUTH_TOKEN": os.environ.get("MODAL_AUTH_TOKEN", "")}
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5.0",
        "torchvision",
        "diffusers>=0.36.0",
        "transformers>=4.55.0",
        "accelerate>=1.2.0",
        "gguf>=0.13.0",
        "safetensors>=0.4.5",
        "sentencepiece",
        "pillow",
        "huggingface_hub",
    )
    .env({"HF_HOME": CACHE_DIR})
    .add_local_python_source("common")
)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("gradio==5.50.0", "fastapi[standard]", "pillow")
    .add_local_python_source("common")
)

app = modal.App(APP_NAME)


@app.cls(
    image=gpu_image,
    gpu=GPU_TYPE,
    volumes={CACHE_DIR: hf_cache},
    secrets=[model_secret],
    timeout=1800,
    scaledown_window=300,
)
class Model:
    @modal.enter()
    def load(self):
        from common.inference import load_pipeline

        self.pipe = load_pipeline()
        hf_cache.commit()  # persist freshly downloaded weights right away

    @modal.method()
    def generate(self, payload: dict) -> dict:
        from common.imgcodec import b64_to_pil, pil_to_b64
        from common.inference import run_edit

        images = [b64_to_pil(s) for s in payload["images"]]
        seed = int(payload.get("seed") or 0)
        outputs = run_edit(
            self.pipe,
            images,
            prompt=payload["prompt"],
            negative_prompt=payload.get("negative_prompt") or " ",
            seed=seed,
            true_cfg_scale=float(payload.get("true_cfg_scale") or 1.0),
            num_inference_steps=int(payload.get("num_inference_steps") or 4),
            num_images=int(payload.get("num_images") or 1),
        )
        return {"images": [pil_to_b64(img) for img in outputs], "seed": seed}


def _validated(payload: dict) -> dict:
    """Normalize an /v1/edit request payload, raising ValueError on bad input."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("'prompt' is required")
    images = payload.get("images") or []
    if not isinstance(images, list) or not images:
        raise ValueError("'images' must be a non-empty list of base64 strings")
    if len(images) > 3:
        raise ValueError("at most 3 input images are supported")
    seed = payload.get("seed")
    if seed is None:
        seed = random.randint(0, MAX_SEED)
    return {
        "prompt": prompt,
        "images": images,
        "negative_prompt": payload.get("negative_prompt") or " ",
        "seed": int(seed),
        "true_cfg_scale": float(payload.get("true_cfg_scale") or 1.0),
        "num_inference_steps": min(max(int(payload.get("num_inference_steps") or 4), 1), 50),
        "num_images": min(max(int(payload.get("num_images") or 1), 1), 4),
    }


@app.function(
    image=web_image,
    secrets=[web_secret],
    timeout=1800,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    import fastapi
    from gradio.routes import mount_gradio_app

    from common.imgcodec import b64_to_pil, pil_to_b64
    from common.ui import build_ui

    auth_token = os.environ.get("MODAL_AUTH_TOKEN", "")
    api = fastapi.FastAPI(title="Qwen Image Edit Rapid AIO GGUF")

    @api.get("/health")
    def health():
        return {"status": "ok"}

    @api.post("/v1/edit")
    def edit(payload: dict, request: fastapi.Request):
        if auth_token:
            if request.headers.get("authorization", "") != f"Bearer {auth_token}":
                raise fastapi.HTTPException(status_code=401, detail="invalid token")
        try:
            payload = _validated(payload)
        except (ValueError, TypeError) as e:
            raise fastapi.HTTPException(status_code=422, detail=str(e))
        return Model().generate.remote(payload)

    def edit_fn(images, prompt, negative_prompt, seed, true_cfg_scale, num_inference_steps, num_images):
        payload = {
            "images": [pil_to_b64(img) for img in images],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "true_cfg_scale": true_cfg_scale,
            "num_inference_steps": num_inference_steps,
            "num_images": num_images,
        }
        result = Model().generate.remote(_validated(payload))
        return [b64_to_pil(s) for s in result["images"]]

    demo = build_ui(edit_fn, subtitle="_Running on [Modal](https://modal.com)._")
    gradio_auth = ("user", auth_token) if auth_token else None
    return mount_gradio_app(api, demo, path="/", auth=gradio_auth)


@app.local_entrypoint()
def warmup():
    """Trigger model download + one tiny generation so real users get warm starts."""
    import base64
    import io

    from PIL import Image

    img = Image.new("RGB", (512, 512), (120, 140, 160))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    result = Model().generate.remote(
        {
            "images": [b64],
            "prompt": "add a small red circle in the center",
            "seed": 0,
            "num_inference_steps": 1,
            "num_images": 1,
        }
    )
    print(f"warmup OK — got {len(result['images'])} image(s) back")
