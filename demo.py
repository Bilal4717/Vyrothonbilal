"""
demo.py — Pocket-Agent Gradio chatbot demo.

Loads the quantized GGUF model and serves a multi-turn chat interface
with visible tool-call output.

Usage:
    python demo.py            # local URL
    python demo.py --share    # public Colab/ngrok URL

Runs on Colab CPU runtime out of the box (no GPU required).
"""

import argparse
import json
import re

import gradio as gr

from inference import run


# ── Formatting helpers ────────────────────────────────────────────────────────

def format_response(raw: str) -> str:
    """Pretty-print <tool_call> blocks; leave plain text as-is."""
    m = re.search(r"<tool_call>(.*?)</tool_call>", raw, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(1).strip())
            pretty = json.dumps(parsed, indent=2)
            return f"🔧 **Tool call detected:**\n```json\n{pretty}\n```"
        except json.JSONDecodeError:
            return f"⚠️ Malformed tool call:\n```\n{raw}\n```"
    return raw


def gradio_to_inference_history(gradio_history: list) -> list:
    """Convert Gradio [(human, bot), …] format to inference.py format."""
    inf_history = []
    for human, bot in gradio_history:
        inf_history.append({"role": "user",      "content": human})
        # Strip markdown formatting when passing back to model
        raw_bot = re.sub(r"🔧 \*\*Tool call detected:\*\*\n```json\n(.*?)\n```",
                         r"<tool_call>\1</tool_call>", bot, flags=re.DOTALL)
        inf_history.append({"role": "assistant", "content": raw_bot})
    return inf_history


# ── Chat function ─────────────────────────────────────────────────────────────

def chat_fn(message: str, history: list) -> str:
    inf_history = gradio_to_inference_history(history)
    raw = run(message, inf_history)
    return format_response(raw)


# ── UI ────────────────────────────────────────────────────────────────────────

DESCRIPTION = """
**Pocket-Agent** — an on-device mobile assistant fine-tuned to call structured tools.

Supported tools: `weather` · `calendar` · `convert` · `currency` · `sql`

For unambiguous requests a JSON tool call appears in a code block.  
For chitchat or impossible tools the model refuses in plain text.
"""

EXAMPLES = [
    ["What's the weather in Paris?"],
    ["100 USD to EUR"],
    ["Convert 50 miles to kilometers"],
    ["Show my calendar for 2025-06-01"],
    ["Create a meeting called Team Sync on 2025-07-15"],
    ["SELECT * FROM orders WHERE status = 'pending'"],
    ["Tell me a joke"],
    ["Book me a flight to Karachi"],
    ["wether in londoon?"],            # typo
    ["Mujhe Paris ka mausam batao"],   # Urdu
    ["¿Cual es el clima en Madrid?"],  # Spanish
]


def build_demo(share: bool = False):
    demo = gr.ChatInterface(
        fn=chat_fn,
        title="🤖 Pocket-Agent — On-Device Tool-Call Assistant",
        description=DESCRIPTION,
        examples=[e[0] for e in EXAMPLES],
        cache_examples=False,
        theme=gr.themes.Soft(),
        chatbot=gr.Chatbot(height=420, render_markdown=True),
        textbox=gr.Textbox(
            placeholder="Ask anything — try 'Weather in Tokyo' or 'Convert 100 miles to km'",
            lines=1,
        ),
    )
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share",  action="store_true", help="Create public share link")
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--server", default="0.0.0.0")
    args = parser.parse_args()

    demo = build_demo()
    demo.launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
        debug=True,
    )


if __name__ == "__main__":
    main()
