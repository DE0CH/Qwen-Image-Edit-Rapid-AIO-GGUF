"""Deployer Space: deploys the Qwen Image Edit backend to Modal.

Runs `modal deploy` + a warm-up generation every time this Space (re)starts,
then stores the endpoint URL on the front-end Space. Progress is pushed to a
private status dataset repo so it can be monitored from anywhere.

Space secrets:   MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, HF_TOKEN,
                 MODAL_AUTH_TOKEN (optional)
Space variables: FRONTEND_SPACE_ID, STATUS_REPO_ID,
                 MODAL_GPU / GGUF_URL / BASE_REPO / LOW_VRAM (optional)
"""

import json
import os
import re
import subprocess
import threading
import time

import gradio as gr
from huggingface_hub import HfApi

LOG_PATH = "/tmp/deploy.log"
STATE = {"status": "starting", "endpoint_url": "", "error": ""}

FRONTEND_SPACE_ID = os.environ.get("FRONTEND_SPACE_ID", "")
STATUS_REPO_ID = os.environ.get("STATUS_REPO_ID", "")
SELF_SPACE_ID = os.environ.get("SPACE_ID", "")

api = HfApi(token=os.environ.get("HF_TOKEN"))


def read_log() -> str:
    try:
        with open(LOG_PATH) as f:
            return f.read()
    except FileNotFoundError:
        return ""


def push_status(status: str, **extra) -> None:
    STATE["status"] = status
    STATE.update(extra)
    print(f"[deployer] {status} {extra}", flush=True)
    if not STATUS_REPO_ID:
        return
    try:
        payload = dict(STATE, ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        api.upload_file(
            path_or_fileobj=json.dumps(payload, indent=2).encode(),
            path_in_repo="status.json",
            repo_id=STATUS_REPO_ID,
            repo_type="dataset",
            commit_message=f"status: {status}",
        )
        api.upload_file(
            path_or_fileobj=read_log().encode(),
            path_in_repo="deploy.log",
            repo_id=STATUS_REPO_ID,
            repo_type="dataset",
            commit_message="deploy log",
        )
    except Exception as e:
        print(f"[deployer] status push failed: {e}", flush=True)


def run_step(name: str, cmd: list[str]) -> bool:
    push_status(name)
    with open(LOG_PATH, "a") as f:
        f.write(f"\n===== {name}: {' '.join(cmd)} =====\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        push_status(f"failed:{name}", error=f"{name} exited with code {proc.returncode}")
        return False
    return True


def run_deploy() -> None:
    if not (os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")):
        push_status("failed:no-credentials", error="MODAL_TOKEN_ID / MODAL_TOKEN_SECRET secrets missing")
        return

    if not run_step("deploying", ["modal", "deploy", "modal_app.py"]):
        return
    match = re.search(r"https://[\w.-]+\.modal\.run", read_log())
    if not match:
        push_status("failed:no-url", error="could not find *.modal.run URL in deploy output")
        return
    url = match.group(0)
    push_status("warming_up", endpoint_url=url)

    if not run_step("warming_up", ["modal", "run", "modal_app.py::warmup"]):
        return

    if FRONTEND_SPACE_ID:
        try:
            api.add_space_variable(FRONTEND_SPACE_ID, "MODAL_ENDPOINT_URL", url)
        except Exception as e:
            push_status("failed:frontend-var", error=f"could not set frontend variable: {e}")
            return
    push_status("done", endpoint_url=url)

    if SELF_SPACE_ID:  # pause so HF restarts don't re-trigger deploys; restart manually to redeploy
        try:
            time.sleep(10)
            api.pause_space(SELF_SPACE_ID)
        except Exception as e:
            print(f"[deployer] self-pause failed (harmless): {e}", flush=True)


threading.Thread(target=run_deploy, daemon=True).start()


def refresh():
    return STATE["status"], STATE.get("endpoint_url", ""), read_log()[-20000:]


with gr.Blocks(title="Modal deployer") as demo:
    gr.Markdown(
        "# Modal deployer\n"
        "Deploys the Qwen Image Edit backend to Modal on every (re)start of this "
        "Space. **Restart this Space to redeploy.**"
    )
    status_box = gr.Textbox(label="Status")
    url_box = gr.Textbox(label="Endpoint URL")
    log_box = gr.Textbox(label="Log", lines=25)
    demo.load(refresh, outputs=[status_box, url_box, log_box])
    timer = gr.Timer(5)
    timer.tick(refresh, outputs=[status_box, url_box, log_box])

demo.launch()
