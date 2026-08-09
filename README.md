# Qwen Image Edit — Rapid AIO Web UI

A web interface (in the style of the official
[Qwen/Qwen-Image-Edit-2511 Space](https://huggingface.co/spaces/Qwen/Qwen-Image-Edit-2511))
for [Phr00t/Qwen-Image-Edit-Rapid-AIO](https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO)
— a Lightning-merged FP8 build of Qwen-Image-Edit-2511 that edits images in
**~4 inference steps** with CFG 1.0. Default checkpoint: **v23 SFW**.

Only the transformer is taken from the AIO checkpoint (loaded with
`QwenImageTransformer2DModel.from_single_file`, FP8 storage with bf16 compute
via layerwise casting); the text encoder / VAE / processor come from the base
`Qwen/Qwen-Image-Edit-2511` repo. No ComfyUI needed. GGUF checkpoints (e.g.
[Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF](https://huggingface.co/Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF))
are also supported via `MODEL_URL`.

## Architecture — two independent deployments

- **Modal** (complete: UI + inference): a GPU backend (default: one L40S,
  scales to zero when idle) serving the Gradio web UI **and** a JSON API at
  `https://<workspace>--qwen-image-edit-rapid-web.modal.run`.
- **Hugging Face Space** (complete: UI + inference): the same Gradio app
  running the model directly on the Space via **ZeroGPU** (requires PRO).
  The Space can instead be deployed as a thin client for the Modal backend
  (`SPACE_MODE=remote`).
- **GitHub Actions** deploys both — everything can be done from a phone
  browser. There is also `scripts/deploy_via_hf.py` for environments where
  the Modal CLI cannot connect (it sets up a private "deployer" Space that
  runs `modal deploy` from Hugging Face infrastructure).

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

When `MODAL_AUTH_TOKEN` is set, the API requires an
`Authorization: Bearer <token>` header, and the Modal-hosted web UI asks for a
login: username `user`, password = the token. The Space authenticates
automatically via its `MODAL_AUTH_TOKEN` secret.

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
| `SPACE_MODE` | `local` | `local` = ZeroGPU inference on the Space; `remote` = thin client for Modal |
| `MODAL_GPU` | `L40S` | e.g. `A100-80GB`, `L4` (with `LOW_VRAM=1`) |
| `MODEL_URL` | Phr00t AIO `v23 SFW` | HF blob URL of another checkpoint (.safetensors AIO or .gguf) |
| `BASE_REPO` | `Qwen/Qwen-Image-Edit-2511` | base pipeline repo |
| `LOW_VRAM` | *(unset)* | set to `1` to enable CPU offload on smaller GPUs (Modal) |
| `MODAL_ENDPOINT_URL` | *(unset)* | backend URL for `SPACE_MODE=remote` |

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

- Default model file: `v23/Qwen-Rapid-AIO-SFW-v23.safetensors` (28.4 GB, FP8).
  Phr00t's notes: v19 is best for edit consistency, v23 for prompt adherence;
  each version has SFW and NSFW variants. Switch with the `MODEL_URL` variable.
- Rapid settings: **4–8 steps, True CFG 1.0**, no negative prompt needed.
