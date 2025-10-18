# Import required libraries
import torch  # PyTorch for tensor operations and GPU support
from transformers import AutoModelForCausalLM, AutoTokenizer  # Hugging Face transformers for model loading

# Configuration parameters
model_id = "nanochat-students/d20-chat-transformers"  # Hugging Face model identifier
max_new_tokens = 64  # Maximum number of new tokens to generate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)  # Load tokenizer for text processing
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=False, dtype=torch.bfloat16).to(device)  # Load model with bfloat16 precision for memory efficiency
model.eval()  # Set model to evaluation mode (disables dropout, batch norm updates)

# Define the conversation in the expected format
conversation = [
    {"role": "user", "content": "What is the capital of France?"},  # User message in chat format
]

# Prepare inputs for the model
inputs = tokenizer.apply_chat_template(
    conversation,  # The conversation to format
    add_generation_prompt=True,  # Add special tokens to indicate where generation should start
    tokenize=True,  # Convert text to token IDs
    return_dict=True,  # Return a dictionary instead of just tensor (needed for **inputs unpacking)
    return_tensors="pt"  # Return PyTorch tensors
).to(device)  # Move tensors to the same device as the model (GPU/CPU)

# Generate response from the model
with torch.no_grad():  # Disable gradient computation for inference (saves memory and speeds up)
    outputs = model.generate(
        **inputs,  # Unpack the input dictionary (input_ids, attention_mask, etc.)
        max_new_tokens=max_new_tokens,  # Limit the number of new tokens to generate
    )

# Extract only the newly generated tokens (excluding the original input prompt)
generated_tokens = outputs[0, inputs.input_ids.shape[1]:]  # Slice from after the input length to the end
print(tokenizer.decode(generated_tokens, skip_special_tokens=True))  # Convert token IDs back to readable text
