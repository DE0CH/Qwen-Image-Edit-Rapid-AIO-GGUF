"""Base64 <-> PIL helpers shared by the API client and server."""

import base64
import io

from PIL import Image

MAX_INPUT_SIDE = 2048


def pil_to_b64(img: Image.Image, format: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def b64_to_pil(data: str) -> Image.Image:
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(data)))
    return img.convert("RGB")


def limit_size(img: Image.Image, max_side: int = MAX_INPUT_SIDE) -> Image.Image:
    """Downscale an image so its longest side is at most max_side."""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
