# LLM Related Stuff

A comprehensive playground and laboratory for experimenting with local AI inference, ranging from lightweight Apple Silicon Python scripts (MLX) to full-stack TypeScript applications built on top of `llama.cpp`.

## 📁 Repository Structure

The repository is organized into distinct domains based on the tech stack and model modality:

### 1. `typescript-apps/` (Local llama.cpp Ecosystem)
Full-stack and CLI applications built in Node.js/TypeScript that interface with a local `llama-server`.
- **`llama-dashboard/`**: A sleek React+Vite web UI for real-time inference monitoring, generation controls, and streaming chat.
- **`llama-benchmark-lab/`**: A programmatic CLI tool to measure prompt-eval and token generation speeds (t/s) across different models and GGUF quantizations.
- **`llama-stream/`**: A lightweight server-sent-events (SSE) client for capturing raw streaming tokens.
- **`llama-company-extractor/`**: A structured JSON extractor utilizing Zod validation and a self-repair loop.
- **`local-company-agent/`**: A mini local agent combining LLM extraction and lead scoring.

### 2. `python-vision-models/` (VLMs & OCR)
Python scripts utilizing MLX, Transformers, and Hugging Face to evaluate visual reasoning.
- Contains experiments for Qwen-VL, DeepSeek-VL, Microsoft Florence, and various OCR tasks.

### 3. `python-language-models/` (Text LLMs)
Python scripts for standard text-generation, conversational UI demos, and coding assistants (e.g., Qwencoder, MiniMax, Kimi).

### 4. `assets/`
Test data, reference images (e.g., 1950s teapot reference), and documents used by the scripts.

---

## 🚀 Quickstart

### For TypeScript / Llama.cpp Apps
These apps expect a `llama-server` to be running locally on port 8080.
1. Start your server:
   ```bash
   llama-server -hf ggml-org/Qwen3-0.6B-GGUF --port 8080
   ```
2. Navigate to an app and run it:
   ```bash
   cd typescript-apps/llama-dashboard
   npm install
   npm run dev
   ```

### For Python MLX/HF Scripts
1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements_minimal.txt
   pip install -r requirements_mps.txt
   ```
3. Run an experiment:
   ```bash
   cd python-vision-models
   python kimi_vlm.py
   ```

---

## 🛠️ System Requirements
- **Hardware**: macOS with Apple Silicon (M1/M2/M3/M4) is highly recommended to take advantage of the Metal backend for both `llama.cpp` and `MLX`.
- **Memory**: 16GB+ Unified Memory (32GB+ recommended for larger 7B-14B Vision models).
- **Storage**: Ensure sufficient disk space; downloading GGUF and safetensors weights can quickly consume hundreds of gigabytes.

## 📝 Notes
- The first execution of any Python script will download model weights automatically from Hugging Face.
- If a model requires custom code, ensure `trust_remote_code=True` is set in the script.
- See model-specific licenses in their respective Hugging Face repositories.
