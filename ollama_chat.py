"""
Ollama Chat Studio — Local LLM chat interface powered by Ollama.

Features:
  • Multi-server management (localhost, LAN, custom with persistence)
  • Streaming responses with markdown rendering
  • Chat history save / load / delete
  • Configurable system prompt, temperature, and token limits
"""

import gradio as gr
import requests
import json
import os
import time
from datetime import datetime
from pathlib import Path

# ─── Paths ──────────────────────────────────────────────────────────────────

APP_DIR = Path(__file__).parent
CHAT_HISTORY_DIR = APP_DIR / "saved_chats"
CHAT_HISTORY_DIR.mkdir(exist_ok=True)
SERVERS_FILE = APP_DIR / "servers.json"

# ─── Defaults ───────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gemma2:latest"

BUILTIN_SERVERS = [
    {"label": "Localhost",    "url": "http://localhost:11434"},
    {"label": "LAN Server",  "url": "http://192.168.50.67:11434"},
]

# ─── Server persistence ────────────────────────────────────────────────────

def _load_custom_servers() -> list[dict]:
    """Load user-added servers from disk."""
    if SERVERS_FILE.exists():
        try:
            return json.loads(SERVERS_FILE.read_text("utf-8"))
        except (json.JSONDecodeError, KeyError):
            pass
    return []


def _save_custom_servers(servers: list[dict]):
    SERVERS_FILE.write_text(json.dumps(servers, indent=2), encoding="utf-8")


def _all_servers() -> list[dict]:
    return BUILTIN_SERVERS + _load_custom_servers()


def _server_choices() -> list[str]:
    """Return display strings for the server dropdown."""
    return [f"{s['label']}  —  {s['url']}" for s in _all_servers()]


def _url_from_choice(choice: str) -> str:
    """Extract the URL from a dropdown display string."""
    for s in _all_servers():
        if choice and s["url"] in choice:
            return s["url"]
    # Fallback: first builtin
    return BUILTIN_SERVERS[0]["url"]


# ─── Ollama helpers ─────────────────────────────────────────────────────────

def _ping(base_url: str, timeout: int = 5):
    """Ping an Ollama server. Returns (ok, model_names, latency_ms)."""
    try:
        t0 = time.monotonic()
        resp = requests.get(f"{base_url}/api/tags", timeout=timeout)
        latency = round((time.monotonic() - t0) * 1000)
        if resp.ok:
            models = [m["name"] for m in resp.json().get("models", [])]
            return True, models, latency
    except Exception:
        pass
    return False, [], 0


def fetch_models(server_choice: str) -> list[str]:
    url = _url_from_choice(server_choice)
    ok, models, _ = _ping(url)
    return models if ok else [DEFAULT_MODEL]


def stream_chat(message, history, model, system_prompt, temperature, max_tokens, server_choice):
    """Generator yielding partial assistant responses."""
    base_url = _url_from_choice(server_choice)

    if not _text(message).strip():
        yield ""
        return

    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    for msg in history:
        if msg.get("role") in ("user", "assistant"):
            txt = _text(msg.get("content", ""))
            if txt:
                messages.append({"role": msg["role"], "content": txt})

    messages.append({"role": "user", "content": _text(message)})

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature},
    }
    if max_tokens > 0:
        payload["options"]["num_predict"] = max_tokens

    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, stream=True, timeout=300)
        resp.raise_for_status()

        partial = ""
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    partial += token
                    yield partial
            except json.JSONDecodeError:
                continue

        if not partial:
            yield "(No response received from the model.)"

    except requests.ConnectionError:
        yield "⚠️ **Connection error** — Cannot reach the Ollama server. Is it running?"
    except requests.Timeout:
        yield "⚠️ **Timeout** — The model took too long to respond."
    except Exception as e:
        yield f"⚠️ **Error** — {type(e).__name__}: {e}"


# ─── Chat persistence ──────────────────────────────────────────────────────

def save_chat(history, model):
    if not history:
        return "Nothing to save.", get_saved_chats()

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    first_msg = next((_text(m["content"]) for m in history if m.get("role") == "user"), "chat")
    label = first_msg[:40].replace(" ", "_").replace("/", "-")
    filename = f"{ts}__{label}.json"

    data = {
        "model": model,
        "saved_at": ts,
        "messages": [{"role": m["role"], "content": _text(m["content"])} for m in history],
    }
    (CHAT_HISTORY_DIR / filename).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return f"Chat saved as **{filename}**", get_saved_chats()


def get_saved_chats():
    return [f.name for f in sorted(CHAT_HISTORY_DIR.glob("*.json"), reverse=True)]


def load_chat(filename):
    if not filename:
        return [], DEFAULT_MODEL, "No file selected."

    filepath = CHAT_HISTORY_DIR / filename
    if not filepath.exists():
        return [], DEFAULT_MODEL, "File not found."

    data = json.loads(filepath.read_text(encoding="utf-8"))
    raw = data.get("messages", [])

    history = []
    for m in raw:
        if "role" in m:
            history.append({"role": m["role"], "content": m["content"]})
        else:
            if m.get("user"):
                history.append({"role": "user", "content": m["user"]})
            if m.get("assistant"):
                history.append({"role": "assistant", "content": m["assistant"]})

    model = data.get("model", DEFAULT_MODEL)
    return history, model, f"Loaded **{filename}** ({len(history)} messages)"


def delete_chat(filename):
    if not filename:
        return "No file selected.", get_saved_chats()
    filepath = CHAT_HISTORY_DIR / filename
    if filepath.exists():
        filepath.unlink()
        return f"Deleted **{filename}**", get_saved_chats()
    return "File not found.", get_saved_chats()


# ─── UI callbacks ───────────────────────────────────────────────────────────

def check_connection(server_choice):
    url = _url_from_choice(server_choice)
    ok, models, latency = _ping(url)
    if ok:
        model_list = ", ".join(models) if models else "none found"
        return (
            f'<div style="background:#062d06;border:1px solid #39ff14;border-radius:8px;'
            f'padding:14px 18px;font-family:monospace;">'
            f'<span style="color:#39ff14;font-size:1.05rem;font-weight:700;">'
            f'✅ CONNECTED</span><br>'
            f'<span style="color:#39ff14cc;font-size:0.85rem;">'
            f'{url}  •  {latency}ms</span><br>'
            f'<span style="color:#39ff14aa;font-size:0.8rem;margin-top:4px;display:inline-block;">'
            f'Models: {model_list}</span></div>'
        )
    return (
        f'<div style="background:#2d0606;border:1px solid #ff4444;border-radius:8px;'
        f'padding:14px 18px;font-family:monospace;">'
        f'<span style="color:#ff6666;font-size:1.05rem;font-weight:700;">'
        f'❌ CONNECTION FAILED</span><br>'
        f'<span style="color:#ff6666aa;font-size:0.85rem;">'
        f'{url}</span></div>'
    )


def refresh_models(server_choice):
    models = fetch_models(server_choice)
    default = DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else DEFAULT_MODEL)
    return gr.update(choices=models, value=default)


def add_custom_server(label, url):
    label = (label or "").strip()
    url = (url or "").strip().rstrip("/")
    if not url:
        return "Enter a URL.", gr.update()
    if not label:
        label = url.split("//")[-1].split(":")[0]

    custom = _load_custom_servers()
    # Prevent duplicates by URL
    if any(s["url"] == url for s in custom) or any(s["url"] == url for s in BUILTIN_SERVERS):
        return f"Server **{url}** already exists.", gr.update()

    custom.append({"label": label, "url": url})
    _save_custom_servers(custom)
    choices = _server_choices()
    new_val = choices[-1]  # just-added server
    return f"Added **{label}** ({url})", gr.update(choices=choices, value=new_val)


def remove_custom_server(server_choice):
    url = _url_from_choice(server_choice)
    # Can't remove builtins
    if any(s["url"] == url for s in BUILTIN_SERVERS):
        return "Built-in servers can't be removed.", gr.update()
    custom = _load_custom_servers()
    custom = [s for s in custom if s["url"] != url]
    _save_custom_servers(custom)
    choices = _server_choices()
    return f"Removed server.", gr.update(choices=choices, value=choices[0])


# ─── Gradio 6 text helper ──────────────────────────────────────────────────

def _text(content):
    """Normalize Gradio message content to plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content) if content else ""


# ─── CSS ────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Layout ─────────────────────────────────────────────────── */
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif !important;
}

/* ── Header ─────────────────────────────────────────────────── */
#header-bar {
    background: linear-gradient(135deg, #0c0c1d 0%, #1a1a3e 50%, #0d2847 100%);
    border-radius: 10px;
    padding: 20px 28px;
    margin-bottom: 10px;
    border: 1px solid rgba(255,255,255,0.06);
}
#header-bar * { color: #c8d6e5 !important; }
#app-title {
    font-size: 1.6rem !important; font-weight: 800 !important;
    color: #ffffff !important; margin: 0 !important;
    letter-spacing: -0.5px;
}
#app-subtitle {
    font-size: 0.82rem !important; color: #7e8fa6 !important;
    margin: 2px 0 0 0 !important;
}

/* ── Sidebar ────────────────────────────────────────────────── */
#sidebar-col {
    background: #f7f9fb;
    border: 1px solid #dfe6ed;
    border-radius: 10px;
    padding: 14px;
}

/* ── Chat area ──────────────────────────────────────────────── */
#chatbot {
    border: 1px solid #dfe6ed !important;
    border-radius: 10px !important;
    min-height: 540px !important;
}
#chatbot .message {
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
}

/* ── Message input ──────────────────────────────────────────── */
#msg-input textarea {
    min-height: 80px !important;
    font-size: 1rem !important;
    border-radius: 10px !important;
    padding: 14px !important;
}

/* ── Buttons ────────────────────────────────────────────────── */
.btn-primary {
    background: linear-gradient(135deg, #0c0c1d, #0d2847) !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
}
.btn-primary:hover { opacity: 0.88 !important; }
.btn-secondary {
    background: #edf1f5 !important; color: #334155 !important;
    border: 1px solid #c8d1da !important; border-radius: 8px !important;
    font-weight: 500 !important;
}
.btn-danger {
    background: #fde8e8 !important; color: #991b1b !important;
    border: 1px solid #f5a3a3 !important; border-radius: 8px !important;
    font-weight: 500 !important;
}

/* ── Section labels ─────────────────────────────────────────── */
.section-label {
    font-size: 0.72rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 1px !important;
    color: #64748b !important; margin: 12px 0 4px 2px !important;
}

/* ── Server management accordion ────────────────────────────── */
.server-accordion { border: none !important; background: transparent !important; }
"""

# ─── Build UI ───────────────────────────────────────────────────────────────

def build_app():
    initial_server_choices = _server_choices()
    initial_server = initial_server_choices[0] if initial_server_choices else ""
    initial_models = fetch_models(initial_server)
    default_model = DEFAULT_MODEL if DEFAULT_MODEL in initial_models else (
        initial_models[0] if initial_models else DEFAULT_MODEL
    )

    with gr.Blocks(title="Ollama Chat Studio", css=CUSTOM_CSS, theme=gr.themes.Soft(
        primary_hue="slate", secondary_hue="blue", neutral_hue="slate",
        font=gr.themes.GoogleFont("IBM Plex Sans"),
    )) as app:

        # ── Header ──────────────────────────────────────────────────
        with gr.Row(elem_id="header-bar"):
            with gr.Column(scale=5):
                gr.Markdown("## 🦙 Ollama Chat Studio", elem_id="app-title")
                gr.Markdown("Local LLM conversations — powered by Ollama", elem_id="app-subtitle")
            with gr.Column(scale=1, min_width=180):
                verify_btn = gr.Button("🔗 Check Connection", elem_classes=["btn-secondary"], size="sm")

        # ── Body ────────────────────────────────────────────────────
        with gr.Row():

            # ── Sidebar ─────────────────────────────────────────────
            with gr.Column(scale=1, min_width=280, elem_id="sidebar-col"):

                # Server
                gr.Markdown("SERVER", elem_classes=["section-label"])
                server_dropdown = gr.Dropdown(
                    choices=initial_server_choices,
                    value=initial_server,
                    label="Active Server",
                    interactive=True,
                )

                with gr.Accordion("Add / Remove Server", open=False, elem_classes=["server-accordion"]):
                    new_label = gr.Textbox(label="Label", placeholder="e.g. Office GPU Box", lines=1)
                    new_url = gr.Textbox(label="URL", placeholder="http://192.168.1.100:11434", lines=1)
                    with gr.Row():
                        add_srv_btn = gr.Button("➕ Add", elem_classes=["btn-secondary"], size="sm")
                        rm_srv_btn = gr.Button("➖ Remove Selected", elem_classes=["btn-danger"], size="sm")

                connection_html = gr.HTML("")

                gr.Markdown("---")

                # Model
                gr.Markdown("MODEL", elem_classes=["section-label"])
                model_dropdown = gr.Dropdown(
                    choices=initial_models, value=default_model,
                    label="Model", interactive=True,
                )
                refresh_btn = gr.Button("🔄 Refresh Models", elem_classes=["btn-secondary"], size="sm")

                gr.Markdown("---")

                # Parameters
                gr.Markdown("PARAMETERS", elem_classes=["section-label"])
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    placeholder="e.g. You are a helpful coding assistant…",
                    lines=3,
                )
                temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
                max_tokens = gr.Slider(0, 8192, value=0, step=64, label="Max Tokens (0 = unlimited)")

                gr.Markdown("---")

                # Saved chats
                gr.Markdown("SAVED CHATS", elem_classes=["section-label"])
                save_btn = gr.Button("💾 Save Current Chat", elem_classes=["btn-primary"], size="sm")
                saved_list = gr.Dropdown(
                    choices=get_saved_chats(), label="Select a chat", interactive=True,
                )
                with gr.Row():
                    load_btn = gr.Button("📂 Load", elem_classes=["btn-secondary"], size="sm")
                    delete_btn = gr.Button("🗑️ Delete", elem_classes=["btn-danger"], size="sm")

                status_md = gr.Markdown("")

            # ── Chat column ─────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    height=350,
                    show_label=False,
                    avatar_images=(None, "https://ollama.com/public/ollama.png"),
                    render_markdown=True,
                    layout="panel",
                    type="messages",
                )
                msg_input = gr.Textbox(
                    placeholder="Type your message and press Enter…",
                    show_label=False,
                    lines=3,
                    max_lines=12,
                    container=False,
                    autofocus=True,
                    elem_id="msg-input",
                )
                with gr.Row():
                    submit_btn = gr.Button("Send ➤", elem_classes=["btn-primary"], scale=1, min_width=120)
                    clear_btn = gr.Button("🧹 Clear Chat", elem_classes=["btn-secondary"], scale=1, min_width=120)

        # ── Wiring ──────────────────────────────────────────────────

        def user_message(user_msg, history):
            return "", history + [{"role": "user", "content": user_msg}]

        def bot_response(history, model, system_prompt, temperature, max_tokens, server_choice):
            if not history:
                yield history
                return
            user_msg = next((_text(m["content"]) for m in reversed(history) if m["role"] == "user"), "")
            past = history[:-1]
            history = history + [{"role": "assistant", "content": ""}]
            for partial in stream_chat(user_msg, past, model, system_prompt, temperature, max_tokens, server_choice):
                history[-1]["content"] = partial
                yield history

        chat_inputs = [chatbot, model_dropdown, system_prompt, temperature, max_tokens, server_dropdown]

        # Submit via Enter
        msg_input.submit(
            fn=user_message, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot], queue=False,
        ).then(fn=bot_response, inputs=chat_inputs, outputs=chatbot)

        # Submit via button
        submit_btn.click(
            fn=user_message, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot], queue=False,
        ).then(fn=bot_response, inputs=chat_inputs, outputs=chatbot)

        # Clear
        clear_btn.click(lambda: [], outputs=chatbot, queue=False)

        # Connection
        verify_btn.click(fn=check_connection, inputs=server_dropdown, outputs=connection_html)

        # Model refresh
        refresh_btn.click(fn=refresh_models, inputs=server_dropdown, outputs=model_dropdown)

        # Server management
        add_srv_btn.click(
            fn=add_custom_server, inputs=[new_label, new_url], outputs=[status_md, server_dropdown],
        )
        rm_srv_btn.click(
            fn=remove_custom_server, inputs=server_dropdown, outputs=[status_md, server_dropdown],
        )

        # Auto-refresh models when server changes
        server_dropdown.change(fn=refresh_models, inputs=server_dropdown, outputs=model_dropdown)

        # Save / Load / Delete
        save_btn.click(fn=save_chat, inputs=[chatbot, model_dropdown], outputs=[status_md, saved_list])
        load_btn.click(fn=load_chat, inputs=saved_list, outputs=[chatbot, model_dropdown, status_md])
        delete_btn.click(fn=delete_chat, inputs=saved_list, outputs=[status_md, saved_list])

    return app


# ─── Launch ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    app.queue()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)
