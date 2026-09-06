# Make sure mlx-vlm is installed
# pip install --upgrade mlx-vlm

import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Load the model (requires custom code)
model, processor = load("mlx-community/Kimi-VL-A3B-Thinking-4bit", trust_remote_code=True)
config = load_config("mlx-community/Kimi-VL-A3B-Thinking-4bit")

# Prepare input
image = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
prompt = "Describe this image."

# Apply chat template
formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=1
)

# Generate output
output = generate(model, processor, formatted_prompt, image)

# Print only the generated text when a GenerationResult object is returned.
try:
    print(output.text)
except AttributeError:
    print(output)
