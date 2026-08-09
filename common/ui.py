"""Shared Gradio UI, modeled on the official Qwen/Qwen-Image-Edit-2511 space.

`build_ui(edit_fn)` takes a backend function with the signature:

    edit_fn(images: list[PIL.Image], prompt: str, negative_prompt: str,
            seed: int, true_cfg_scale: float, num_inference_steps: int,
            num_images: int) -> list[PIL.Image]

so the same UI can run against a local pipeline or a remote (Modal) API.
"""

import random

import gradio as gr

MAX_SEED = 2**31 - 1
MAX_IMAGES_IN = 3

# iOS Safari zooms into a focused input when its font-size is below 16px.
CSS = """
input[type='text'], input[type='number'], textarea, select {
    font-size: 16px !important;
}
"""

DESCRIPTION = """
Fast **4-step** image editing with
[Qwen-Image-Edit-Rapid-AIO (GGUF)](https://huggingface.co/Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF)
— a Lightning-merged build of
[Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511).
Upload 1–3 images, describe the edit, and hit **Edit**.
"""


def _to_pil_list(gallery):
    images = []
    for item in gallery or []:
        img = item[0] if isinstance(item, (tuple, list)) else item
        if img is not None:
            images.append(img)
    return images[:MAX_IMAGES_IN]


def build_ui(edit_fn, subtitle: str = "") -> gr.Blocks:
    def run(
        gallery,
        prompt,
        negative_prompt,
        seed,
        randomize_seed,
        true_cfg_scale,
        num_inference_steps,
        num_images,
        progress=gr.Progress(track_tqdm=True),
    ):
        images = _to_pil_list(gallery)
        if not images:
            raise gr.Error("Please upload at least one input image.")
        if not prompt or not prompt.strip():
            raise gr.Error("Please describe the edit you want.")
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        try:
            outputs = edit_fn(
                images=images,
                prompt=prompt.strip(),
                negative_prompt=(negative_prompt or "").strip() or " ",
                seed=int(seed),
                true_cfg_scale=float(true_cfg_scale),
                num_inference_steps=int(num_inference_steps),
                num_images=int(num_images),
            )
        except gr.Error:
            raise
        except Exception as e:  # surface backend errors in the UI
            raise gr.Error(f"Generation failed: {e}")
        return outputs, seed

    with gr.Blocks(
        title="Qwen Image Edit — Rapid AIO GGUF", theme=gr.themes.Soft(), css=CSS
    ) as demo:
        gr.Markdown("# 🖌️ Qwen Image Edit — Rapid AIO (GGUF)")
        gr.Markdown(DESCRIPTION + (f"\n\n{subtitle}" if subtitle else ""))

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
        run_button.click(fn=run, inputs=inputs, outputs=[result, seed])
        prompt.submit(fn=run, inputs=inputs, outputs=[result, seed])

    demo.queue(max_size=32, default_concurrency_limit=8)
    return demo
