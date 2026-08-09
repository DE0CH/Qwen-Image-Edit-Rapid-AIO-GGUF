---
title: Qwen Image Edit Rapid AIO GGUF
emoji: 🖌️
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Fast 4-step image editing (Qwen-Image-Edit Rapid AIO GGUF)
---

# Qwen Image Edit — Rapid AIO (GGUF)

Web UI for [Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF](https://huggingface.co/Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF),
a Lightning-merged GGUF build of [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
that edits images in ~4 inference steps.

Inference runs directly on this Space via ZeroGPU: the GGUF transformer is
loaded with diffusers' GGUF support and the text encoder/VAE come from the
base Qwen/Qwen-Image-Edit-2511 repo.

Model overrides via Space variables: `GGUF_URL` (blob URL of another .gguf
quant), `BASE_REPO`. If `MODAL_ENDPOINT_URL` is set instead, the app switches
to thin-client mode and calls that backend rather than running locally.

Deployment source:
[DE0CH/Qwen-Image-Edit-Rapid-AIO-GGUF](https://github.com/DE0CH/Qwen-Image-Edit-Rapid-AIO-GGUF)
