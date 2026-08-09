"""Deploy the Gradio app to a Hugging Face Space.

Two modes (SPACE_MODE env var):
  local  (default) - fully self-contained Space: runs GGUF inference on the
                     Space itself via ZeroGPU (requires a PRO account).
                     Removes any MODAL_* wiring from the Space.
  remote           - thin front-end on free CPU hardware that calls a Modal
                     backend (MODAL_ENDPOINT_URL required).

Environment variables:
  HF_TOKEN            (required) write-scoped Hugging Face token
  HF_SPACE_ID         (optional) e.g. "username/qwen-image-edit-rapid";
                      defaults to "<token owner>/qwen-image-edit-rapid"
  SPACE_MODE          (optional) "local" (default) or "remote"
  MODAL_ENDPOINT_URL  (remote mode) Modal backend URL; stored as Space variable
  MODAL_AUTH_TOKEN    (remote mode, optional) bearer token; stored as secret
  GGUF_URL/BASE_REPO  (local mode, optional) model overrides; stored as variables
"""

import os
import shutil
import sys
import tempfile

from huggingface_hub import HfApi

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZERO_GPU_HARDWARE = "zero-a10g"


def build_local_requirements() -> str:
    with open(os.path.join(REPO_ROOT, "requirements.txt")) as f:
        reqs = f.read()
    return reqs + "spaces\n"


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set")
    mode = os.environ.get("SPACE_MODE", "local").strip().lower()
    if mode not in ("local", "remote"):
        sys.exit(f"unknown SPACE_MODE {mode!r}")
    api = HfApi(token=token)

    space_id = os.environ.get("HF_SPACE_ID", "").strip()
    if not space_id:
        space_id = f"{api.whoami()['name']}/qwen-image-edit-rapid"

    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        private=os.environ.get("SPACE_PUBLIC", "") != "1",
    )

    with tempfile.TemporaryDirectory() as build_dir:
        shutil.copy(os.path.join(REPO_ROOT, "app.py"), build_dir)
        shutil.copytree(
            os.path.join(REPO_ROOT, "common"),
            os.path.join(build_dir, "common"),
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copy(
            os.path.join(REPO_ROOT, "space", "README.md"),
            os.path.join(build_dir, "README.md"),
        )
        if mode == "local":
            with open(os.path.join(build_dir, "requirements.txt"), "w") as f:
                f.write(build_local_requirements())
        else:
            shutil.copy(
                os.path.join(REPO_ROOT, "space", "requirements.txt"),
                os.path.join(build_dir, "requirements.txt"),
            )
        api.upload_folder(
            folder_path=build_dir,
            repo_id=space_id,
            repo_type="space",
            commit_message=f"Deploy from source repo ({mode} mode)",
        )

    if mode == "local":
        # Make sure the app doesn't fall back to remote mode.
        for var in ("MODAL_ENDPOINT_URL",):
            try:
                api.delete_space_variable(space_id, var)
            except Exception:
                pass
        for secret in ("MODAL_AUTH_TOKEN",):
            try:
                api.delete_space_secret(space_id, secret)
            except Exception:
                pass
        for key in ("MODEL_URL", "GGUF_URL", "BASE_REPO"):
            if os.environ.get(key):
                api.add_space_variable(space_id, key, os.environ[key])
        # Authenticated HF downloads (faster, higher rate limits)
        api.add_space_secret(space_id, "HF_TOKEN", token)
        api.request_space_hardware(space_id, ZERO_GPU_HARDWARE)
        print(f"Requested hardware: {ZERO_GPU_HARDWARE}")
    else:
        endpoint_url = os.environ.get("MODAL_ENDPOINT_URL", "").strip()
        if endpoint_url:
            api.add_space_variable(space_id, "MODAL_ENDPOINT_URL", endpoint_url)
            print(f"Set Space variable MODAL_ENDPOINT_URL={endpoint_url}")
        else:
            print("WARNING: remote mode but no MODAL_ENDPOINT_URL set")
        auth_token = os.environ.get("MODAL_AUTH_TOKEN", "").strip()
        if auth_token:
            api.add_space_secret(space_id, "MODAL_AUTH_TOKEN", auth_token)
            print("Set Space secret MODAL_AUTH_TOKEN")

    try:
        api.restart_space(space_id)
    except Exception as e:  # a Space mid-build can refuse restarts; not fatal
        print(f"restart_space skipped: {e}")

    print(f"Space deployed ({mode} mode): https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
