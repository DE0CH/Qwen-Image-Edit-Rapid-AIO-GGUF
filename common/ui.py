"""Shared Gradio UI, modeled on the official Qwen/Qwen-Image-Edit-2511 space.

`build_ui(edit_fn)` takes a backend function with the signature:

    edit_fn(images: list[PIL.Image], prompt: str, negative_prompt: str,
            seed: int, true_cfg_scale: float, num_inference_steps: int,
            num_images: int) -> list[PIL.Image]

so the same UI can run against a local pipeline or a remote (Modal) API.

## Surviving a disconnect while rendering

Gradio has no native job-resume (tracked upstream as gradio-app/gradio#13584,
slated for Gradio 7): when the browser's SSE connection to `/queue/data` drops
— which is exactly what a phone does when you switch apps or lock the screen —
Gradio calls `clean_events()` and the in-flight generation is thrown away, so
the result never comes back.

We work around it entirely at the app level:

  * The generation runs in a **background thread** we own, not inside the
    Gradio event. Gradio cannot cancel it, and on ZeroGPU a `@spaces.GPU` call
    made off the request thread is scheduled with `request=None`, so ZeroGPU
    does NOT wire up the connection-liveness watchdog that would otherwise kill
    the GPU job when the client disconnects (see spaces/zero/client.py).
  * The result is stashed in a process-wide cache keyed by a random `job_id`.
  * That `job_id` is persisted in the browser via `gr.BrowserState`
    (localStorage), so it survives a reload, a tab switch, or the page being
    frozen and thawed.
  * A `gr.Timer` polls the cache while a job is running, and `demo.load`
    reattaches to an in-flight (or just-finished) job when the page comes back.

So you can hit Edit, switch away, and the result is waiting when you return.
"""

import random
import threading
import time
import uuid

import gradio as gr

MAX_SEED = 2**31 - 1
MAX_IMAGES_IN = 3

# How long a finished job stays in the cache so a returning client can still
# pick it up. Running jobs are never pruned.
JOB_TTL_SECONDS = 3600
POLL_INTERVAL_SECONDS = 1.5

# Process-wide job store: job_id -> {status, images, seed, error, updated}.
# status is one of "running" | "done" | "error".
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()

# iOS Safari zooms into a focused input when its font-size is below 16px.
CSS = """
input[type='text'], input[type='number'], textarea, select {
    font-size: 16px !important;
}
"""

DESCRIPTION = """
Fast **4-step** image editing with
[Qwen-Image-Edit-Rapid-AIO](https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO)
— a Lightning-merged build of
[Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511).
Upload 1–3 images, describe the edit, and hit **Edit**. You can switch away
while it renders — the result is kept and shown when you come back.
"""


def _to_pil_list(gallery):
    images = []
    for item in gallery or []:
        img = item[0] if isinstance(item, (tuple, list)) else item
        if img is not None:
            images.append(img)
    return images[:MAX_IMAGES_IN]


def _set_job(job_id: str, **fields) -> None:
    with _JOBS_LOCK:
        job = _JOBS.setdefault(job_id, {})
        job.update(fields)
        job["updated"] = time.time()


def _get_job(job_id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def _prune_jobs() -> None:
    now = time.time()
    with _JOBS_LOCK:
        stale = [
            jid
            for jid, job in _JOBS.items()
            if job.get("status") in ("done", "error")
            and now - job.get("updated", 0) > JOB_TTL_SECONDS
        ]
        for jid in stale:
            _JOBS.pop(jid, None)


def build_ui(edit_fn, subtitle: str = "") -> gr.Blocks:
    def _worker(job_id, images, prompt, negative_prompt, seed,
                true_cfg_scale, num_inference_steps, num_images):
        """Run the (potentially GPU-bound) edit off the request thread."""
        try:
            outputs = edit_fn(
                images=images,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                true_cfg_scale=true_cfg_scale,
                num_inference_steps=num_inference_steps,
                num_images=num_images,
            )
            _set_job(job_id, status="done", images=outputs, seed=seed, error=None)
        except gr.Error as e:
            _set_job(job_id, status="error", error=str(e.message))
        except Exception as e:  # surface backend errors in the UI
            _set_job(job_id, status="error", error=f"Generation failed: {e}")

    def submit(
        gallery,
        prompt,
        negative_prompt,
        seed,
        randomize_seed,
        true_cfg_scale,
        num_inference_steps,
        num_images,
    ):
        """Validate, kick off a detached job, and hand its id to the browser."""
        images = _to_pil_list(gallery)
        if not images:
            raise gr.Error("Please upload at least one input image.")
        if not prompt or not prompt.strip():
            raise gr.Error("Please describe the edit you want.")
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)

        _prune_jobs()
        job_id = uuid.uuid4().hex
        _set_job(job_id, status="running", images=None, seed=int(seed), error=None)

        thread = threading.Thread(
            target=_worker,
            args=(
                job_id,
                images,
                prompt.strip(),
                (negative_prompt or "").strip() or " ",
                int(seed),
                float(true_cfg_scale),
                int(num_inference_steps),
                int(num_images),
            ),
            daemon=True,
        )
        thread.start()

        # Outputs: job_state (-> localStorage), poll timer on, status, clear result.
        return (
            job_id,
            gr.Timer(active=True),
            gr.update(
                value="⏳ **Generating…** you can switch away — the result will be "
                "waiting here when you come back."
            ),
            None,
        )

    def _render(job_id, *, on_load: bool):
        """Shared logic for the poll timer and the reattach-on-load handler.

        Returns updates for [result, seed, timer, status].
        """
        keep = gr.update()
        if not job_id:
            return keep, keep, gr.Timer(active=False), keep

        job = _get_job(job_id)
        if job is None:
            # Unknown id (expired, or the Space restarted and lost the cache).
            msg = "" if not on_load else ""
            return keep, keep, gr.Timer(active=False), gr.update(value=msg)

        status = job.get("status")
        if status == "running":
            note = "⏳ **Still generating…** hang tight." if on_load else keep
            return keep, keep, gr.Timer(active=True), note
        if status == "error":
            return (
                keep,
                keep,
                gr.Timer(active=False),
                gr.update(value=f"❌ {job.get('error', 'Generation failed.')}"),
            )
        # done
        done_note = "✅ Restored your last result." if on_load else ""
        return (
            job.get("images"),
            job.get("seed", keep),
            gr.Timer(active=False),
            gr.update(value=done_note),
        )

    def poll(job_id):
        return _render(job_id, on_load=False)

    def reattach(job_id):
        return _render(job_id, on_load=True)

    with gr.Blocks(
        title="Qwen Image Edit — Rapid AIO GGUF", theme=gr.themes.Soft(), css=CSS
    ) as demo:
        gr.Markdown("# 🖌️ Qwen Image Edit — Rapid AIO (GGUF)")
        gr.Markdown(DESCRIPTION + (f"\n\n{subtitle}" if subtitle else ""))

        # Persisted across reloads / tab switches (localStorage).
        job_state = gr.BrowserState("", storage_key="qwen_image_edit_rapid_job")
        # Polls the job cache while a job is running; toggled active/inactive.
        poll_timer = gr.Timer(POLL_INTERVAL_SECONDS, active=False)

        with gr.Row():
            with gr.Column(scale=1):
                input_gallery = gr.Gallery(
                    label=f"Input images (up to {MAX_IMAGES_IN})",
                    type="pil",
                    columns=3,
                    rows=1,
                    height=280,
                    interactive=True,
                )
                prompt = gr.Textbox(
                    label="Edit instruction",
                    placeholder="e.g. Change the background to a snowy mountain at sunset",
                    lines=2,
                )
                run_button = gr.Button("Edit", variant="primary")
                with gr.Accordion("Advanced settings", open=False):
                    negative_prompt = gr.Textbox(
                        label="Negative prompt (used when CFG > 1)",
                        placeholder="What to avoid",
                        lines=1,
                    )
                    with gr.Row():
                        seed = gr.Slider(
                            label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0
                        )
                        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    with gr.Row():
                        true_cfg_scale = gr.Slider(
                            label="True CFG scale (Rapid model: keep at 1.0)",
                            minimum=1.0,
                            maximum=10.0,
                            step=0.1,
                            value=1.0,
                        )
                        num_inference_steps = gr.Slider(
                            label="Inference steps (Rapid model: 4–8)",
                            minimum=1,
                            maximum=28,
                            step=1,
                            value=4,
                        )
                    num_images = gr.Slider(
                        label="Number of output images",
                        minimum=1,
                        maximum=4,
                        step=1,
                        value=1,
                    )
            with gr.Column(scale=1):
                status = gr.Markdown("")
                result = gr.Gallery(
                    label="Result", columns=2, height=420, format="png"
                )

        inputs = [
            input_gallery,
            prompt,
            negative_prompt,
            seed,
            randomize_seed,
            true_cfg_scale,
            num_inference_steps,
            num_images,
        ]
        submit_outputs = [job_state, poll_timer, status, result]
        render_outputs = [result, seed, poll_timer, status]

        run_button.click(fn=submit, inputs=inputs, outputs=submit_outputs)
        prompt.submit(fn=submit, inputs=inputs, outputs=submit_outputs)
        poll_timer.tick(fn=poll, inputs=[job_state], outputs=render_outputs)
        # On (re)load, restore/attach to whatever job the browser remembers.
        demo.load(fn=reattach, inputs=[job_state], outputs=render_outputs)

    demo.queue(max_size=32, default_concurrency_limit=8)
    return demo
