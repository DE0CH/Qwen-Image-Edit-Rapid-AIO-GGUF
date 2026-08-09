"""Gradio entrypoint for Hugging Face Spaces (or local use).

Two modes:
  * MODAL_ENDPOINT_URL set  -> thin client that calls the Modal backend API.
    Runs fine on free CPU Space hardware; only needs gradio/requests/pillow.
    Optional MODAL_AUTH_TOKEN is sent as a Bearer token.
  * MODAL_ENDPOINT_URL unset -> loads the pipeline locally (needs a GPU and
    the full requirements.txt). On a ZeroGPU Space the edit function is
    automatically wrapped with @spaces.GPU.
"""

import os

from common.ui import build_ui

MODAL_ENDPOINT_URL = os.environ.get("MODAL_ENDPOINT_URL", "").strip().rstrip("/")


def make_remote_edit_fn():
    import requests

    from common.imgcodec import b64_to_pil, pil_to_b64

    token = os.environ.get("MODAL_AUTH_TOKEN", "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def edit_fn(images, prompt, negative_prompt, seed, true_cfg_scale, num_inference_steps, num_images):
        payload = {
            "images": [pil_to_b64(img.convert("RGB")) for img in images],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "true_cfg_scale": true_cfg_scale,
            "num_inference_steps": num_inference_steps,
            "num_images": num_images,
        }
        resp = requests.post(
            f"{MODAL_ENDPOINT_URL}/v1/edit", json=payload, headers=headers, timeout=1800
        )
        if resp.status_code != 200:
            raise RuntimeError(f"backend returned {resp.status_code}: {resp.text[:300]}")
        return [b64_to_pil(s) for s in resp.json()["images"]]

    return edit_fn


def make_local_edit_fn():
    from common.inference import load_pipeline, run_edit

    pipe = load_pipeline()

    def edit_fn(images, prompt, negative_prompt, seed, true_cfg_scale, num_inference_steps, num_images):
        return run_edit(
            pipe,
            images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            true_cfg_scale=true_cfg_scale,
            num_inference_steps=num_inference_steps,
            num_images=num_images,
        )

    try:  # ZeroGPU Spaces: allocate a GPU per call
        import spaces

        edit_fn = spaces.GPU(duration=120)(edit_fn)
    except ImportError:
        pass
    return edit_fn


if MODAL_ENDPOINT_URL:
    demo = build_ui(
        make_remote_edit_fn(),
        subtitle="_Inference runs on a [Modal](https://modal.com) GPU backend._",
    )
else:
    demo = build_ui(make_local_edit_fn())

if __name__ == "__main__":
    demo.launch()
