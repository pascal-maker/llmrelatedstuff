import gradio as gr
from transformers import pipeline

# 1. Build the VQA pipeline
pipe = pipeline(
    task="image-text-to-text",
    model="Qwen/Qwen2-VL-2B",          # or another compatible model
    device_map="auto",                 # put on GPU if available
    torch_dtype="auto"                 # uses fp16 on modern GPUs
)

# 2. Gradio callback
def answer_question(image, question):
    """
    image   : PIL image (Gradio supplies it in this format)
    question: str
    """
    if image is None or question.strip() == "":
        return "Please provide both an image and a question 😊"

    # Feed a *single* dict as the pipeline expects
    result = pipe({"image": image, "text": question})

    # The pipeline returns a dictionary (not a list) for one example
    return result["generated_text"]

# 3. Build & launch the UI
gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Image(type="pil", label="Upload an image"),
        gr.Textbox(lines=2,
                   label="Ask a question about the image",
                   placeholder="e.g. What animal is on the candy?")
    ],
    outputs="text",
    title="Visual Q&A with Qwen2-VL-2B",
    description=(
        "Upload an image and ask a question about it. "
        "The Qwen2-VL-2B model will answer using multimodal reasoning."
    ),
).launch(share=False)   # set share=True if you want a public link
