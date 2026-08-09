"""Deploy the Gradio front-end to a Hugging Face Space.

Environment variables:
  HF_TOKEN            (required) write-scoped Hugging Face token
  HF_SPACE_ID         (optional) e.g. "username/qwen-image-edit-rapid";
                      defaults to "<token owner>/qwen-image-edit-rapid"
  MODAL_ENDPOINT_URL  (optional) Modal backend URL; stored as a Space variable
  MODAL_AUTH_TOKEN    (optional) backend bearer token; stored as a Space secret
"""

import os
import shutil
import sys
import tempfile

from huggingface_hub import HfApi

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set")
    api = HfApi(token=token)

    space_id = os.environ.get("HF_SPACE_ID", "").strip()
    if not space_id:
        space_id = f"{api.whoami()['name']}/qwen-image-edit-rapid"

    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
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
        shutil.copy(
            os.path.join(REPO_ROOT, "space", "requirements.txt"),
            os.path.join(build_dir, "requirements.txt"),
        )
        api.upload_folder(
            folder_path=build_dir,
            repo_id=space_id,
            repo_type="space",
            commit_message="Deploy from GitHub Actions",
        )

    endpoint_url = os.environ.get("MODAL_ENDPOINT_URL", "").strip()
    if endpoint_url:
        api.add_space_variable(space_id, "MODAL_ENDPOINT_URL", endpoint_url)
        print(f"Set Space variable MODAL_ENDPOINT_URL={endpoint_url}")
    else:
        print(
            "WARNING: no MODAL_ENDPOINT_URL available - set it manually in the "
            "Space settings or the UI will try to run inference locally (no GPU on CPU Spaces)."
        )

    auth_token = os.environ.get("MODAL_AUTH_TOKEN", "").strip()
    if auth_token:
        api.add_space_secret(space_id, "MODAL_AUTH_TOKEN", auth_token)
        print("Set Space secret MODAL_AUTH_TOKEN")

    try:
        api.restart_space(space_id)
    except Exception as e:  # a Space mid-build can refuse restarts; not fatal
        print(f"restart_space skipped: {e}")

    print(f"Space deployed: https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
