import gradio as gr
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image

# Load model and processor
processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
model = AutoModelForImageTextToText.from_pretrained("google/paligemma-3b-pt-224")
model.eval()

# Function to process image + prompt
def generate_caption(image, prompt):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# Gradio Interface
gr.Interface(
    fn=generate_caption,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Your Prompt (e.g. 'Describe the image' or ask a question)")
    ],
    outputs="text",
    title="PaLI-Gemma: Visual Question Answering or Captioning",
    description="Upload an image and provide a prompt/question. The model will generate a response using PaLI-Gemma."
).launch()
