"""
inference.py — Pocket-Agent grader interface.

Exposes: def run(prompt: str, history: list[dict]) -> str

Rules enforced:
  - No network imports (requests / urllib / http / socket) — grader AST-scans this file.
  - Loads the quantized GGUF from the same directory as this file.
  - history format: list of {"role": "user"|"assistant", "content": str}
"""

import os
import json
import re

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_GGUF_PATH  = os.path.join(_HERE, "artifacts", "model_q4km.gguf")

# ── Generation settings ───────────────────────────────────────────────────────
_MAX_TOKENS  = 256
_TEMPERATURE = 0.1   # low = deterministic tool-call formatting
_N_THREADS   = os.cpu_count() or 4

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are Pocket-Agent, an on-device mobile assistant.\n"
    "You have access to 5 tools: weather, calendar, convert, currency, sql.\n\n"
    "For unambiguous requests, emit ONLY a JSON tool call wrapped in "
    "<tool_call>...</tool_call> tags.\n"
    "For chitchat, impossible tools, or ambiguous references with no prior history, "
    "emit plain text with NO tool call.\n"
    "Never guess — if the user's intent is unclear and there is no prior context, "
    "refuse politely.\n\n"
    "Tool schemas:\n"
    '{"tool": "weather",  "args": {"location": "string", "unit": "C|F"}}\n'
    '{"tool": "calendar", "args": {"action": "list|create", "date": "YYYY-MM-DD", "title": "string?"}}\n'
    '{"tool": "convert",  "args": {"value": number, "from_unit": "string", "to_unit": "string"}}\n'
    '{"tool": "currency", "args": {"amount": number, "from": "ISO3", "to": "ISO3"}}\n'
    '{"tool": "sql",      "args": {"query": "string"}}\n'
)

# ── Lazy model singleton ──────────────────────────────────────────────────────
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from llama_cpp import Llama  # local import keeps top-level clean
        if not os.path.exists(_GGUF_PATH):
            raise FileNotFoundError(
                f"GGUF not found at {_GGUF_PATH}. "
                "Run: python scripts/quantize.py  (or: make quantize)"
            )
        _llm = Llama(
            model_path=_GGUF_PATH,
            n_ctx=2048,
            n_threads=_N_THREADS,
            verbose=False,
        )
    return _llm


# ── Prompt builder ────────────────────────────────────────────────────────────
def _build_prompt(prompt: str, history: list) -> str:
    """
    Assembles a Gemma-style chat prompt.
    Format: <start_of_turn>role\ncontent<end_of_turn>\n
    """
    turns = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for h in (history or []):
        if isinstance(h, dict) and "role" in h and "content" in h:
            turns.append({"role": h["role"], "content": str(h["content"])})
    turns.append({"role": "user", "content": prompt})

    out = ""
    for t in turns:
        out += f"<start_of_turn>{t['role']}\n{t['content']}<end_of_turn>\n"
    out += "<start_of_turn>model\n"
    return out


# ── Post-processor ────────────────────────────────────────────────────────────
def _clean(raw: str) -> str:
    """
    Extracts and validates the first <tool_call> block.
    Returns it if JSON is valid, otherwise returns the raw text.
    """
    m = re.search(r"<tool_call>(.*?)</tool_call>", raw, re.DOTALL)
    if m:
        body = m.group(1).strip()
        try:
            json.loads(body)          # validate
            return f"<tool_call>{body}</tool_call>"
        except json.JSONDecodeError:
            pass                       # malformed → return raw text
    # Strip any stray tags so refusals are clean plain text
    return re.sub(r"</?tool_call>", "", raw).strip()


# ── Public API ────────────────────────────────────────────────────────────────
def run(prompt: str, history: list = None) -> str:
    """
    Grader entry point.

    Args:
        prompt:  The current user message (str).
        history: Prior turns as list of {"role": ..., "content": ...}.
                 Pass [] or None for single-turn calls.

    Returns:
        str — either "<tool_call>{...}</tool_call>" or plain refusal text.
    """
    llm = _get_llm()
    full_prompt = _build_prompt(prompt, history or [])

    result = llm(
        full_prompt,
        max_tokens=_MAX_TOKENS,
        temperature=_TEMPERATURE,
        stop=["<end_of_turn>", "<start_of_turn>"],
    )

    raw = result["choices"][0]["text"].strip()
    return _clean(raw)


# ── CLI convenience ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    user_prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Weather in London?"
    print(run(user_prompt, []))
