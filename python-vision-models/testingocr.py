#!/usr/bin/env python3
"""
EasyOCR Gradio demo (CPU | CUDA)  – safe on Apple-Silicon

• Languages default to English + Dutch
• If user selects "mps", we fall back to CPU and show a notice
• detail = 0 → plain text list
  detail = 1 → annotated image + raw result
"""

import platform
from functools import lru_cache
from typing import List, Tuple, Union

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

import easyocr

# ───────────────────────── available devices ────────────────────────────
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICES.append("mps")          # will fall back internally

# ───────────────────────── language list ────────────────────────────────
try:
    from easyocr.languages import LANGUAGES as _LANGS_DICT
    AVAILABLE_LANGS = sorted(_LANGS_DICT.keys())
except Exception:
    AVAILABLE_LANGS = ["en", "nl"]

# ───────────────────────── cached Reader helper ──────────────────────────
@lru_cache(maxsize=8)
def make_reader(langs: Tuple[str, ...], true_device: str) -> easyocr.Reader:
    """Return an EasyOCR.Reader on CPU or CUDA."""
    reader = easyocr.Reader(list(langs), gpu=(true_device == "cuda"))
    return reader


# ───────────────────────── OCR inference fn ──────────────────────────────
def run_easyocr(
    image: Union[Image.Image, np.ndarray],
    langs: List[str],
    device: str,
    detail: int,
):
    # transparent MPS-to-CPU fallback
    true_device = "cpu" if device == "mps" else device
    fallback_note = ("⚠️  MPS not yet supported by EasyOCR – running on CPU instead.\n\n"
                     if device == "mps" else "")

    if isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
    else:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    reader = make_reader(tuple(langs), true_device)

    with torch.inference_mode():
        if true_device == "cuda":
            torch.cuda.empty_cache()
        result = reader.readtext(img_rgb, detail=detail, batch_size=1)

    # ---- output ---------------------------------------------------------
    if detail == 0:
        return None, fallback_note + "\n".join(result)

    boxed = img_rgb.copy()
    for bbox, txt, conf in result:
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(boxed, [pts], True, (0, 255, 0), 2)
        cv2.putText(
            boxed,
            f"{txt} ({conf:.2f})",
            (pts[0][0], pts[0][1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return Image.fromarray(boxed), fallback_note + str(result)


# ──────────────────────────── Gradio UI ───────────────────────────────────
DESCRIPTION = (
    "### 📜 EasyOCR demo  \n"
    f"Host: **{platform.system()} {platform.machine()}**\n\n"
    "CUDA is fully supported. Apple-Silicon MPS will automatically fall back to CPU "
    "because EasyOCR hasn’t enabled MPS yet."
)

with gr.Blocks(title="EasyOCR demo") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(label="Upload image", type="pil", height=260)

            lang_sel = gr.Dropdown(
                AVAILABLE_LANGS,
                value=["en", "nl"],
                multiselect=True,
                label="Languages",
            )

            dev_sel = gr.Radio(
                DEVICES,
                value=DEVICES[-1],          # show MPS if present, else CPU
                label="Device (MPS auto-falls back)",
            )

            detail_sel = gr.Radio(
                [0, 1],
                value=1,
                label="Detail",
                info="0 = text only,  1 = draw bounding boxes",
            )

            run_btn = gr.Button("Run OCR", variant="primary")

        with gr.Column(scale=2):
            img_out = gr.Image(label="Annotated image (detail 1)", height=260)
            txt_out = gr.Textbox(label="OCR output", lines=12)

    run_btn.click(
        fn=run_easyocr,
        inputs=[img_in, lang_sel, dev_sel, detail_sel],
        outputs=[img_out, txt_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True, inbrowser=True)
