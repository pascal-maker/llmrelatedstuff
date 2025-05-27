"""Florence-2 + TrOCR Object Detection and OCR
Detects objects in images and reads any text inside each box.
"""

import torch
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)
import os

# ─────────────────────────────────────────────────────────────
# Device & dtype (Fixed for MPS compatibility)
# ─────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device, torch_dtype = "mps", torch.float32  # Use float32 instead of float16 for MPS
    # Set MPS fallback for unsupported operations
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
elif torch.cuda.is_available():
    device, torch_dtype = "cuda:0", torch.float16
else:
    device, torch_dtype = "cpu", torch.float32

print(f"Using device: {device} with dtype: {torch_dtype}")

# ─────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────
FLORENCE_ID = "microsoft/Florence-2-large"
TROCR_ID = "microsoft/trocr-base-handwritten"

florence = AutoModelForCausalLM.from_pretrained(
    FLORENCE_ID, torch_dtype=torch_dtype, trust_remote_code=True
).to(device)

florence_proc = AutoProcessor.from_pretrained(FLORENCE_ID, trust_remote_code=True)

# Load TrOCR with direct processor and model for better control
trocr_processor = TrOCRProcessor.from_pretrained(TROCR_ID)
trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_ID)

# Move TrOCR to appropriate device
if device.startswith(("cuda", "mps")):
    trocr_model = trocr_model.to(device)
    if device.startswith("mps"):
        trocr_model = trocr_model.to(torch.float32)  # Ensure float32 for MPS

# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────
def _extract_boxes_labels(det):
    if "bboxes" in det:
        labels = det.get("labels") or det.get("class_names")
        return det["bboxes"], labels
    if "quad_boxes" in det:  # OCR_WITH_REGION format
        # Convert quad boxes (8 coords) to regular boxes (4 coords)
        boxes = []
        for quad in det["quad_boxes"]:
            # quad format: [x1,y1,x2,y2,x3,y3,x4,y4] -> convert to [xmin,ymin,xmax,ymax]
            x_coords = [quad[i] for i in range(0, 8, 2)]
            y_coords = [quad[i] for i in range(1, 8, 2)]
            xmin, xmax = min(x_coords), max(x_coords)
            ymin, ymax = min(y_coords), max(y_coords)
            boxes.append([xmin, ymin, xmax, ymax])
        labels = det.get("labels") or ["text"] * len(boxes)
        return boxes, labels
    if "instances" in det:  # rare alt format
        boxes = [i["bbox"] for i in det["instances"]]
        labels = [i["label"] for i in det["instances"]]
        return boxes, labels
    raise ValueError("Florence output missing bboxes / labels")


def _text_size(draw, text, font):
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    return r - l, b - t


def annotate(image, boxes, labels, texts):
    im = image.convert("RGB").copy()
    draw = ImageDraw.Draw(im, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=14)
    except OSError:
        font = ImageFont.load_default()

    for (x0, y0, x1, y1), label, text in zip(boxes, labels, texts):
        draw.rectangle((x0, y0, x1, y1), outline="red", width=3)

        caption = f"{label}: {text[:40]}…" if text else label
        tw, th = _text_size(draw, caption, font)
        draw.rectangle((x0, y0 - th, x0 + tw, y0), fill="red")
        draw.text((x0, y0 - th), caption, fill="white", font=font)
    return im


# ─────────────────────────────────────────────────────────────
# Gradio callback
# ─────────────────────────────────────────────────────────────
def preprocess_crop_for_ocr(crop):
    """Enhanced preprocessing for better handwritten OCR"""
    from PIL import ImageEnhance, ImageOps, ImageFilter
    import numpy as np
    
    # Convert to RGB if not already (TrOCR expects RGB)
    if crop.mode != 'RGB':
        crop = crop.convert('RGB')
    
    # Resize to optimal size for TrOCR (384x384 is the training size)
    # But maintain aspect ratio
    w, h = crop.size
    max_size = 384
    
    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)
    
    # Ensure minimum size
    if new_w < 32:
        new_w = 32
    if new_h < 32:
        new_h = 32
        
    crop = crop.resize((new_w, new_h), Image.LANCZOS)
    
    # Create versions with different enhancements
    versions = [crop]  # Original
    
    # Version 1: Enhanced contrast and sharpness
    enhanced = crop.copy()
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = enhancer.enhance(1.3)
    versions.append(enhanced)
    
    # Version 2: Convert to grayscale then back to RGB with high contrast
    gray_version = crop.convert('L')
    gray_version = ImageOps.autocontrast(gray_version, cutoff=2)
    gray_version = gray_version.convert('RGB')
    versions.append(gray_version)
    
    return versions

def run_trocr_on_crop(crop_versions):
    """Run TrOCR on multiple versions of the same crop and return best result"""
    results = []
    
    for i, crop in enumerate(crop_versions):
        try:
            # Process image
            pixel_values = trocr_processor(images=crop, return_tensors="pt").pixel_values
            
            # Move to appropriate device
            if device.startswith(("cuda", "mps")):
                pixel_values = pixel_values.to(device)
                if device.startswith("mps"):
                    pixel_values = pixel_values.to(torch.float32)
            
            # Generate text with better parameters for handwriting
            with torch.inference_mode():
                generated_ids = trocr_model.generate(
                    pixel_values,
                    max_length=256,  # Allow longer text
                    num_beams=5,     # More beams for better results
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=trocr_processor.tokenizer.pad_token_id,
                )
            
            generated_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            results.append((generated_text.strip(), len(generated_text.strip())))
            
        except Exception as e:
            print(f"TrOCR error on version {i}: {e}")
            results.append(("", 0))
    
    # Return the longest non-empty result
    valid_results = [(text, length) for text, length in results if text and length > 1]
    if valid_results:
        return max(valid_results, key=lambda x: x[1])[0]
    else:
        return results[0][0] if results else ""

def analyse(image: Image.Image):
    if image is None:
        return None, "Upload an image first."

    # Try multiple Florence tasks for better detection
    tasks_to_try = ["<OCR_WITH_REGION>", "<OD>", "<DENSE_REGION_CAPTION>"]
    
    best_det = None
    best_boxes = []
    best_labels = []
    
    for task in tasks_to_try:
        try:
            inputs = florence_proc(
                text=task, images=image, return_tensors="pt"
            ).to(device, torch_dtype)

            with torch.inference_mode():
                gen_ids = florence.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                )

            gen_txt = florence_proc.batch_decode(gen_ids, skip_special_tokens=False)[0]
            det = florence_proc.post_process_generation(
                gen_txt, task=task, image_size=image.size
            )
            det = det.get(task, det)  # unwrap if nested
            
            print(f"Task {task} result keys: {det.keys() if isinstance(det, dict) else 'Not a dict'}")
            
            if det and ("bboxes" in det or "quad_boxes" in det or "instances" in det):
                boxes, labels = _extract_boxes_labels(det)
                if boxes and len(boxes) > len(best_boxes):
                    best_boxes, best_labels = boxes, labels
                    print(f"Task {task} found {len(boxes)} regions")
                    if task == "<OCR_WITH_REGION>":
                        break  # Prefer OCR_WITH_REGION if it finds anything
        except Exception as e:
            print(f"Task {task} failed: {e}")
            continue
    
    boxes, labels = best_boxes, best_labels
    
    if not boxes:
        return image, "No text regions detected. Try a clearer image with better lighting."

    # Run OCR per box with enhanced preprocessing and direct TrOCR
    texts = []
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = map(int, box)
        
        # Add some padding around detected regions
        padding = 15  # Increased padding
        x0 = max(0, x0 - padding)
        y0 = max(0, y0 - padding)
        x1 = min(image.width, x1 + padding)
        y1 = min(image.height, y1 + padding)
        
        crop = image.crop((x0, y0, x1, y1))
        
        # Get multiple enhanced versions of the crop
        crop_versions = preprocess_crop_for_ocr(crop)
        
        try:
            # Run TrOCR on all versions and get best result
            text = run_trocr_on_crop(crop_versions)
            print(f"Box {i}: '{text}'")
            
        except Exception as e:
            print(f"OCR error for box {i}: {e}")
            text = ""
        texts.append(text)

    annotated_img = annotate(image, boxes, labels, texts)

    # build structured JSON result
    result = [
        {"bbox": box, "label": label, "text": text}
        for box, label, text in zip(boxes, labels, texts)
    ]
    return annotated_img, result


# ─────────────────────────────────────────────────────────────
# Interface
# ─────────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=analyse,
    inputs=gr.Image(type="pil", label="Upload image"),
    outputs=[
        gr.Image(type="pil", label="Detections + OCR"),
        gr.JSON(label="Details (bbox, label, text)"),
    ],
    title="Florence-2 + TrOCR  •  Object Detection + Reading",
    description=(
        "Florence-2 finds objects and text regions; TrOCR reads handwritten and printed text. "
        "Enhanced for better envelope and document reading with multiple detection methods."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(share=True)