# 🦙 Ollama Chat Studio

A local LLM chat interface powered by [Ollama](https://ollama.com/) — fast, private, and completely free.

Built with Python and [Gradio](https://www.gradio.app/), this app gives you a clean, full-featured chat UI for any model running on your local machine or LAN. No API keys, no cloud, no subscriptions.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Gradio](https://img.shields.io/badge/Gradio-5.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Why This Exists

Cloud LLMs are great, but they cost money, require internet, and send your data to someone else's servers. With Ollama and the right model, you can run a shockingly capable AI assistant on your own hardware — and with Google's **Gemma 4 E4B**, the quality gap between local and paid has nearly disappeared.

This repo pairs a polished Gradio chat app with a step-by-step guide to get Gemma 4 E4B running at **100% GPU** on an NVIDIA card.

---

## Features

- **Streaming responses** with full markdown rendering
- **Multi-server management** — localhost, LAN, or custom Ollama endpoints (with persistence)
- **Chat history** — save, load, and delete conversations
- **Configurable parameters** — system prompt, temperature, max tokens
- **Auto-detect models** — pulls the model list from whichever Ollama server you connect to
- **One-click launchers** — batch file (Windows) and shell script (macOS/Linux)

---

## Recommended Model: Gemma 4 E4B

[**Gemma 4 E4B**](https://ollama.com/library/gemma4:e4b) is Google DeepMind's latest open model, released under the Apache 2.0 license. The "E4B" stands for "Effective 4B" — it's a Mixture-of-Experts architecture with ~8B total parameters but only 4B active at inference time. The result is frontier-class reasoning, coding, and multimodal capabilities that fit comfortably on a single consumer GPU.

On an **RTX 4070 (12 GB VRAM)**, this model runs incredibly fast with the Modelfile configuration below — responses stream in near-instantly and the quality rivals paid services I used to pay for.

---

## Setup Guide

### 1. Install Ollama

Download and install Ollama for your platform:

👉 **[https://ollama.com/download](https://ollama.com/download)**

After installation, confirm it's working by opening a terminal and running:

```bash
ollama --version
```

### 2. Pull the Gemma 4 E4B Model

```bash
ollama pull gemma4:e4b
```

This downloads the Q4_K_M quantized model (~9.6 GB). Once complete, you can test it immediately:

```bash
ollama run gemma4:e4b
```

### 3. Create a Custom Modelfile for Full GPU Offload

> **This is the critical step.** By default, Ollama may only offload part of the model to your GPU. On my RTX 4070, the base `gemma4:e4b` was running at roughly **64% GPU / 32% CPU** — noticeably slower with CPU bottlenecking the generation.
>
> After creating a Modelfile with `num_gpu 999` and a 32K context window, it jumped to **100% GPU utilization** and the speed difference was dramatic.

Create a file called `Modelfile` anywhere on your system with the following contents:

```
FROM gemma4:e4b
PARAMETER num_gpu 999
PARAMETER num_ctx 32768
```

Then build the custom model:

```bash
ollama create gemma4-e4b-32k -f Modelfile
```

This creates a new model variant called `gemma4-e4b-32k` that forces all layers onto the GPU and sets a 32K token context window. You can verify it was created:

```bash
ollama list
```

You should see `gemma4-e4b-32k` alongside the base `gemma4:e4b`. From now on, select `gemma4-e4b-32k` in the chat app for the best performance.

### 4. Install Python Dependencies

> **Requires Python 3.10 or higher.** Download from [python.org](https://www.python.org/downloads/) if you don't have it.

Clone this repository, then install dependencies:

```bash
git clone https://github.com/Cosmos-Guru/Ollama-Chat-Studio.git
cd Ollama-Chat-Studio
pip install -r requirements.txt
```

Or let the launcher scripts handle it automatically (they create a virtual environment on first run).

### 5. Launch the App

**Windows:**

Double-click `run_ollama_chat.bat` or run it from a terminal:

```cmd
run_ollama_chat.bat
```

**macOS / Linux:**

```bash
chmod +x run_ollama_chat.sh
./run_ollama_chat.sh
```

The app opens automatically in your browser at **http://localhost:7860**.

---

## GPU Offload: Before & After

| Configuration | GPU Usage | CPU Usage | Speed |
|---|---|---|---|
| `gemma4:e4b` (default) | ~64% | ~32% | Moderate — CPU bottleneck visible |
| `gemma4-e4b-32k` (Modelfile) | **100%** | Minimal | **Blazing fast** — streams instantly |

The `num_gpu 999` parameter tells Ollama to offload every layer to the GPU. The `num_ctx 32768` sets a generous 32K context window so longer conversations don't get truncated. If you have a GPU with less VRAM (8 GB), you can try reducing `num_ctx` to `16384` or `8192`.

---

## Project Structure

```
Ollama-Chat-Studio/
├── ollama_chat.py          # Main application
├── requirements.txt        # Python dependencies
├── run_ollama_chat.bat     # Windows launcher (auto-setup on first run)
├── run_ollama_chat.sh      # macOS/Linux launcher (auto-setup on first run)
├── Modelfile               # Ollama Modelfile for gemma4-e4b-32k
├── saved_chats/            # Auto-created — your conversation history
├── servers.json            # Auto-created — custom server list
└── README.md
```

---

## Configuration

Once the app is running, everything is configurable from the sidebar:

- **Server** — Switch between localhost, LAN machines, or add custom Ollama endpoints
- **Model** — Select any model available on the connected server
- **System Prompt** — Set persistent instructions for the model
- **Temperature** — Control randomness (0.0 = deterministic, 2.0 = creative)
- **Max Tokens** — Cap response length (0 = unlimited)

---

## Troubleshooting

**"Connection error — Cannot reach the Ollama server"**
Make sure Ollama is running. On Windows, check the system tray. On macOS/Linux, run `ollama serve` in a separate terminal.

**Model not appearing in the dropdown**
Click "Refresh Models" in the sidebar. If you just created the Modelfile variant, it may take a moment to appear.

**Slow generation despite GPU**
You're probably running the base model without the Modelfile. Follow Step 3 above to create the `gemma4-e4b-32k` variant and select it in the app.

**VRAM out of memory**
Reduce `num_ctx` in the Modelfile. Try `16384` or `8192` and recreate the model with `ollama create`.

---

## Links

- **Ollama** — [https://ollama.com](https://ollama.com)
- **Ollama Download** — [https://ollama.com/download](https://ollama.com/download)
- **Gemma 4 E4B on Ollama** — [https://ollama.com/library/gemma4:e4b](https://ollama.com/library/gemma4:e4b)
- **Gemma 4 by Google DeepMind** — [https://deepmind.google/models/gemma/gemma-4/](https://deepmind.google/models/gemma/gemma-4/)
- **Gradio** — [https://www.gradio.app](https://www.gradio.app)
- **Python** — [https://www.python.org/downloads/](https://www.python.org/downloads/)

---

## License

MIT — do whatever you want with it.
