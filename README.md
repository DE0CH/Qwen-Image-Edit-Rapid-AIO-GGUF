# Qwen Image Edit — Rapid AIO (GGUF) Web UI

A web interface (in the style of the official
[Qwen/Qwen-Image-Edit-2511 Space](https://huggingface.co/spaces/Qwen/Qwen-Image-Edit-2511))
for [Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF](https://huggingface.co/Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF)
— a Lightning-merged GGUF build of Qwen-Image-Edit-2511 that edits images in
**~4 inference steps** with CFG 1.0.

The GGUF file only contains the diffusion transformer, so this project loads it
with `diffusers` (`QwenImageTransformer2DModel.from_single_file` +
`GGUFQuantizationConfig`) and pulls the text encoder / VAE / processor from the
base `Qwen/Qwen-Image-Edit-2511` repo. No ComfyUI needed.

## Architecture

- **Modal** hosts the GPU backend (default: one L40S, scales to zero when idle)
  and serves **both** a Gradio web UI and a JSON API at
  `https://<workspace>--qwen-image-edit-rapid-web.modal.run`.
- **Hugging Face Space** hosts the same Gradio UI as a lightweight front-end on
  free CPU hardware, calling the Modal API for inference.
- **GitHub Actions** deploys both — everything can be done from a phone browser.

## Deploy (works entirely from a phone)

### 1. Get tokens

- **Modal**: sign up at [modal.com](https://modal.com), then go to
  **Settings → API Tokens → New Token**. Copy the token ID (`ak-…`) and secret
  (`as-…`).
- **Hugging Face**: [Settings → Access Tokens](https://huggingface.co/settings/tokens)
  → create a token with **Write** access.

### 2. Add them to this GitHub repo

**Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Value |
|---|---|
| `MODAL_TOKEN_ID` | `ak-…` |
| `MODAL_TOKEN_SECRET` | `as-…` |
| `HF_TOKEN` | your write token |
| `MODAL_AUTH_TOKEN` | *(optional)* any password — protects the backend API & Modal UI |

### 3. Run the workflow

**Actions tab → Deploy → Run workflow** (it also runs automatically on every
push to `main`). The job summary shows both URLs when done:

- Modal web UI + API: `https://<workspace>--qwen-image-edit-rapid-web.modal.run`
- Space: `https://huggingface.co/spaces/<you>/qwen-image-edit-rapid`

The first deploy includes a warm-up run that downloads ~35 GB of weights into a
Modal volume (can take 15–30 min). After that, cold starts are much faster and
the container scales to zero after 5 idle minutes, so you only pay while
generating.

### Optional repository *variables*

**Settings → Secrets and variables → Actions → Variables**:

| Variable | Default | Purpose |
|---|---|---|
| `HF_SPACE_ID` | `<you>/qwen-image-edit-rapid` | Space name |
| `MODAL_GPU` | `L40S` | e.g. `A100-80GB`, `L4` (with `LOW_VRAM=1`) |
| `GGUF_URL` | v9.0 `Q4_K_M` | HF blob URL of any other .gguf from the model repo |
| `BASE_REPO` | `Qwen/Qwen-Image-Edit-2511` | base pipeline repo |
| `LOW_VRAM` | *(unset)* | set to `1` to enable CPU offload on smaller GPUs |
| `MODAL_ENDPOINT_URL` | *(from Modal deploy)* | manual backend URL override for the Space |

## API

```bash
curl -X POST "https://<workspace>--qwen-image-edit-rapid-web.modal.run/v1/edit" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MODAL_AUTH_TOKEN" \
  -d '{
        "prompt": "make the sky a dramatic sunset",
        "images": ["<base64-encoded png or jpeg>"],
        "num_inference_steps": 4,
        "true_cfg_scale": 1.0
      }'
```

Response: `{"images": ["<base64 png>", …], "seed": 123}`. Up to 3 input images
and 4 outputs per request. Omit `seed` for a random one.

## Run locally (GPU with ~40 GB VRAM, or less with `LOW_VRAM=1`)

```bash
pip install -r requirements.txt
python app.py            # local inference
# or use a remote backend:
MODAL_ENDPOINT_URL=https://…modal.run python app.py
```

## Notes

- Default model file: `v90/qwen-rapid-nsfw-v9.0-Q4_K_M.gguf` (13.3 GB) — the
  latest version in the model repo (v9.0 only ships under that filename).
  Switch quants/versions with the `GGUF_URL` variable.
- Rapid settings: **4–8 steps, True CFG 1.0**, no negative prompt needed.
- Alternative to Modal: give the Space **ZeroGPU** hardware (PRO account),
  remove the `MODAL_ENDPOINT_URL` variable, and replace the Space's
  `requirements.txt` with the root one from this repo — `app.py` then runs
  inference on the Space itself.
