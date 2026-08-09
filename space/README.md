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

This Space is a lightweight front-end: inference runs on a [Modal](https://modal.com)
GPU backend. Configure it in the Space settings:

- **Variable `MODAL_ENDPOINT_URL`** (required): the Modal web endpoint URL,
  e.g. `https://<workspace>--qwen-image-edit-rapid-web.modal.run`
- **Secret `MODAL_AUTH_TOKEN`** (optional): bearer token if the backend was
  deployed with authentication enabled

Deployment source and instructions:
[DE0CH/Qwen-Image-Edit-Rapid-AIO-GGUF](https://github.com/DE0CH/Qwen-Image-Edit-Rapid-AIO-GGUF)
