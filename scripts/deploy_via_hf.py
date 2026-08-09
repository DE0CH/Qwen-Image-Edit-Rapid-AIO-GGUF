"""Set up the private 'deployer' Space that deploys the backend to Modal.

For use where direct `modal deploy` is impossible (e.g. gRPC-blocked networks):
the deployer Space runs it from Hugging Face infrastructure instead.

Environment variables:
  HF_TOKEN            (required) write-scoped Hugging Face token
  MODAL_TOKEN_ID      (required) ak-...
  MODAL_TOKEN_SECRET  (required) as-...
  MODAL_AUTH_TOKEN    (optional) protect the backend with a bearer token
  DEPLOYER_SPACE_ID   (optional) deployer Space id
  STATUS_REPO_ID      (optional) private dataset repo for status reporting
  MODAL_GPU / GGUF_URL / BASE_REPO / LOW_VRAM  (optional) backend config
"""

import os
import shutil
import sys
import tempfile

from huggingface_hub import HfApi

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN")
    token_id = os.environ.get("MODAL_TOKEN_ID")
    token_secret = os.environ.get("MODAL_TOKEN_SECRET")
    if not (hf_token and token_id and token_secret):
        sys.exit("HF_TOKEN, MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are required")

    api = HfApi(token=hf_token)
    user = api.whoami()["name"]
    deployer_id = os.environ.get("DEPLOYER_SPACE_ID") or f"{user}/modal-deployer-qwen-edit"
    status_repo = os.environ.get("STATUS_REPO_ID") or f"{user}/modal-deploy-status"

    api.create_repo(status_repo, repo_type="dataset", private=True, exist_ok=True)
    api.create_repo(
        deployer_id, repo_type="space", space_sdk="gradio", private=True, exist_ok=True
    )

    api.add_space_secret(deployer_id, "MODAL_TOKEN_ID", token_id)
    api.add_space_secret(deployer_id, "MODAL_TOKEN_SECRET", token_secret)
    api.add_space_secret(deployer_id, "HF_TOKEN", hf_token)
    if os.environ.get("MODAL_AUTH_TOKEN"):
        api.add_space_secret(deployer_id, "MODAL_AUTH_TOKEN", os.environ["MODAL_AUTH_TOKEN"])
    api.add_space_variable(deployer_id, "STATUS_REPO_ID", status_repo)
    for key in ("MODAL_GPU", "MODEL_URL", "GGUF_URL", "BASE_REPO", "LOW_VRAM"):
        if os.environ.get(key):
            api.add_space_variable(deployer_id, key, os.environ[key])

    deployer_src = os.path.join(REPO_ROOT, "scripts", "hf_modal_deployer")
    with tempfile.TemporaryDirectory() as build_dir:
        for name in os.listdir(deployer_src):
            src = os.path.join(deployer_src, name)
            if os.path.isfile(src):
                shutil.copy(src, build_dir)
        shutil.copy(os.path.join(REPO_ROOT, "modal_app.py"), build_dir)
        shutil.copytree(
            os.path.join(REPO_ROOT, "common"),
            os.path.join(build_dir, "common"),
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        api.upload_folder(
            folder_path=build_dir,
            repo_id=deployer_id,
            repo_type="space",
            commit_message="Deploy backend to Modal",
        )

    try:
        api.restart_space(deployer_id)
    except Exception as e:
        print(f"restart_space skipped: {e}")

    print(f"deployer: https://huggingface.co/spaces/{deployer_id}")
    print(f"status:   https://huggingface.co/datasets/{status_repo}")


if __name__ == "__main__":
    main()
