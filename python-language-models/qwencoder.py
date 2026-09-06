import os

from mlx_lm import load, generate

# Enable the faster multi-connection downloader if installed.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
try:
    import hf_transfer  # noqa: F401
except Exception:
    print('Tip: install hf_transfer for faster downloads: pip install -U "huggingface_hub[hf-transfer]"')

model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-4bit")

prompt = "write a story about elon musk and jack ma discussing the future of AI"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=False,
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens=512)
