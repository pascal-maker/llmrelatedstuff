# Make sure mlx-vlm is installed
# pip install --upgrade mlx-vlm

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Load the model
model, processor = load("mlx-community/DeepSeek-OCR-8bit")
config = load_config("mlx-community/DeepSeek-OCR-8bit")

# Prepare input
image = ["woman-in-a-1950s-dress-holding-a-teapot-D4FWTT.jpg"]
prompt = "read what is on the image written."

# Apply chat template
formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=1
)

# Generate output
output = generate(model, processor, formatted_prompt, image)
print(output)