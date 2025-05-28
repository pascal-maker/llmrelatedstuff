import gradio as gr
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

"""Gradio demo for Microsoft TrOCR‑Base (handwritten) OCR"""

MODEL_NAME = "microsoft/trocr-base-handwritten"

# --- Load processor & model --------------------------------------------------
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# Use GPU if available for faster generation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)
model.eval()

# --- OCR inference function --------------------------------------------------

def ocr_handwritten(img: Image.Image) -> str:
    """Return handwritten text recognised in the provided image."""
    if img is None:
        return ""

    # Ensure RGB input expected by processor
    if img.mode != "RGB":
        img = img.convert("RGB")

    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)

    # Autoregressive generation
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# --- Gradio interface --------------------------------------------------------

demo = gr.Interface(
    fn=ocr_handwritten,
    inputs=gr.Image(type="pil", label="Upload an image with handwritten text"),
    outputs=gr.Textbox(label="Recognised text"),
    title="Handwritten OCR with TrOCR (base handwritten)",
    description=(
        "Upload or drag‑and‑drop an image and get the OCR result using "
        "Microsoft's TrOCR base handwritten model."
    ),
)

if __name__ == "__main__":
    demo.launch()
