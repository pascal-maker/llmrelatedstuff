# llmrelatedstuff

Practical experiments and mini‑demos for LLMs, VLMs, OCR, and multimodal models using Hugging Face, Transformers, and MLX.

**Highlights**
- Lightweight MLX demos for Apple Silicon
- Multimodal scripts for image/video/audio understanding
- Quick experiments with popular open models

**Requirements**
- Python 3.10+
- macOS + Apple Silicon recommended for MLX scripts
- Enough disk for model downloads (many models are multi‑GB)

**Quickstart**
1. Create and activate a virtual environment.
2. Install dependencies as needed for the script you run.

Common installs:
```
pip install -r requirements_minimal.txt
```
```
pip install -r requirements_mps.txt
```
```
pip install --upgrade mlx-lm mlx-vlm
```

**Examples**

Text generation (MLX):
```
python kimik2.py
```

Vision-language (MLX‑VLM):
```
python kimi_vlm.py
```

**Notes**
- First run will download model weights and may take a while.
- If a model requires custom code, you may need `trust_remote_code=True`.

**Repository Layout**
- `kimik2.py`: MLX‑LM text generation demo.
- `kimi_vlm.py`: MLX‑VLM image captioning demo.
- `omnivinci/`: research code for an omni‑modal model (large and GPU‑oriented).
- `notebooks/`: exploratory notebooks.

**License**
See model‑specific licenses in their respective repos. This repo is for open experimentation and examples.
