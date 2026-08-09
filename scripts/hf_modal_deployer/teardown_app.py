"""Teardown runner: stops ALL deployed Modal apps and deletes this project's
cache volume. Used in place of the deployer's app.py for decommissioning.

Reports results to the status dataset repo as teardown.json.
"""

import json
import os
import subprocess
import threading
import time

import gradio as gr
from huggingface_hub import HfApi

STATUS_REPO_ID = os.environ.get("STATUS_REPO_ID", "")
OWN_VOLUME = "qwen-image-edit-rapid-hf-cache"
LOG_PATH = "/tmp/teardown.log"

api = HfApi(token=os.environ.get("HF_TOKEN"))
STATE = {
    "status": "starting",
    "apps_stopped": [],
    "apps_failed": [],
    "volumes_deleted": [],
    "volumes_left": [],
    "error": "",
}


def log(msg: str) -> None:
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def read_log() -> str:
    try:
        with open(LOG_PATH) as f:
            return f.read()
    except FileNotFoundError:
        return ""


def push_status(status: str, **extra) -> None:
    STATE["status"] = status
    STATE.update(extra)
    if not STATUS_REPO_ID:
        return
    try:
        payload = dict(STATE, ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        api.upload_file(
            path_or_fileobj=json.dumps(payload, indent=2).encode(),
            path_in_repo="teardown.json",
            repo_id=STATUS_REPO_ID,
            repo_type="dataset",
            commit_message=f"teardown: {status}",
        )
        api.upload_file(
            path_or_fileobj=read_log().encode(),
            path_in_repo="teardown.log",
            repo_id=STATUS_REPO_ID,
            repo_type="dataset",
            commit_message="teardown log",
        )
    except Exception as e:
        print(f"status push failed: {e}", flush=True)


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True)
    log(f"$ {' '.join(cmd)}\n{p.stdout}{p.stderr}")
    return p


def field(row: dict, *names):
    for n in names:
        for key, value in row.items():
            if key.lower().replace(" ", "_") == n:
                return value
    return None


def teardown() -> None:
    p = run(["modal", "app", "list", "--json"])
    if p.returncode != 0:
        push_status("failed:list-apps", error=p.stderr[-500:])
        return
    try:
        apps = json.loads(p.stdout)
    except json.JSONDecodeError:
        push_status("failed:parse-apps", error=p.stdout[-500:])
        return

    stopped, failed = [], []
    for app_row in apps:
        state = str(field(app_row, "state") or "").lower()
        if "stopped" in state:
            continue
        app_id = field(app_row, "app_id", "id")
        name = field(app_row, "description", "name") or app_id
        r = run(["modal", "app", "stop", str(app_id)])
        (stopped if r.returncode == 0 else failed).append(f"{name} ({app_id})")
        push_status("stopping-apps", apps_stopped=stopped, apps_failed=failed)

    deleted, left = [], []
    v = run(["modal", "volume", "list", "--json"])
    if v.returncode == 0:
        try:
            volumes = json.loads(v.stdout)
        except json.JSONDecodeError:
            volumes = []
        for vol in volumes:
            vname = field(vol, "name")
            if vname == OWN_VOLUME:
                r = run(["modal", "volume", "delete", vname, "--yes"])
                (deleted if r.returncode == 0 else left).append(vname)
            elif vname:
                left.append(vname)

    status = "done" if not failed else "done-with-failures"
    push_status(status, apps_stopped=stopped, apps_failed=failed,
                volumes_deleted=deleted, volumes_left=left)


threading.Thread(target=teardown, daemon=True).start()


def refresh():
    return STATE["status"], read_log()[-20000:]


with gr.Blocks(title="Modal teardown") as demo:
    gr.Markdown("# Modal teardown\nStops all deployed Modal apps and deletes the cache volume.")
    status_box = gr.Textbox(label="Status")
    log_box = gr.Textbox(label="Log", lines=25)
    demo.load(refresh, outputs=[status_box, log_box])
    timer = gr.Timer(5)
    timer.tick(refresh, outputs=[status_box, log_box])

demo.launch()
