"""
eval_harness_contract.py
========================
This file defines the EXACT interface the grader uses to score your submission.
Do NOT modify this file. Your inference.py must satisfy this contract.

Grader calls:
    from inference import run
    score = grade_example(run(prompt, history), expected_output)

Scoring rules (per example):
    +1.0  Exact tool match, all args correct (numerical args within ±1%)
    +0.5  Correct tool, ≥1 arg wrong
     0.0  Wrong tool, malformed JSON, or wrong refusal decision
    -0.5  Emitted a tool call when a refusal was the correct answer
"""

import json
import re
import sys
import importlib
from pathlib import Path


# ── Tool-call extraction ──────────────────────────────────────────────────────

def extract_tool_call(text: str) -> dict | None:
    """
    Returns the parsed tool-call dict from a <tool_call>...</tool_call> block,
    or None if no valid tool call is present.
    """
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except (json.JSONDecodeError, ValueError):
        return None


# ── Argument comparison ───────────────────────────────────────────────────────

def args_equal(pred: dict, gold: dict, tol: float = 0.01) -> bool:
    """
    Returns True iff every key in gold is present in pred with a matching value.
    Numerical values are compared within ±tol (relative).
    String values are compared case-insensitively after stripping.
    """
    if set(pred.keys()) != set(gold.keys()):
        return False
    for key, gval in gold.items():
        pval = pred.get(key)
        if isinstance(gval, (int, float)):
            if not isinstance(pval, (int, float)):
                return False
            denom = max(abs(float(gval)), 1e-9)
            if abs(float(pval) - float(gval)) / denom > tol:
                return False
        else:
            if str(pval).strip().lower() != str(gval).strip().lower():
                return False
    return True


# ── Per-example scorer ────────────────────────────────────────────────────────

def grade_example(prediction: str, expected: str) -> tuple[float, str]:
    """
    Returns (score, reason) where score ∈ {-0.5, 0.0, 0.5, 1.0}.

    Args:
        prediction: Raw string returned by run(prompt, history).
        expected:   Gold-standard string (either a <tool_call>…</tool_call>
                    or a plain-text refusal marker "__REFUSAL__").
    """
    pred_tc = extract_tool_call(prediction)
    gold_tc = extract_tool_call(expected)

    gold_is_refusal = gold_tc is None
    pred_is_refusal = pred_tc is None

    # ── Refusal cases ─────────────────────────────────────────────────────────
    if gold_is_refusal:
        if pred_is_refusal:
            return 1.0, "correct refusal"
        else:
            return -0.5, f"tool call emitted when refusal expected (tool={pred_tc.get('tool')})"

    # ── Gold expects a tool call ──────────────────────────────────────────────
    if pred_is_refusal:
        return 0.0, "refusal emitted when tool call expected"

    if pred_tc is None:
        return 0.0, "malformed JSON in <tool_call>"

    pred_tool = pred_tc.get("tool", "")
    gold_tool = gold_tc.get("tool", "")

    if pred_tool != gold_tool:
        return 0.0, f"wrong tool: got '{pred_tool}', expected '{gold_tool}'"

    pred_args = pred_tc.get("args", {})
    gold_args = gold_tc.get("args", {})

    if args_equal(pred_args, gold_args):
        return 1.0, "exact match"
    else:
        return 0.5, f"correct tool '{gold_tool}', args mismatch — pred={pred_args} gold={gold_args}"


# ── Batch evaluation ──────────────────────────────────────────────────────────

def evaluate(test_file: str, inference_module_path: str = ".") -> dict:
    """
    Runs the full evaluation loop against a JSONL test file.

    JSONL format (one JSON object per line):
        {
          "id": "A-001",
          "slice": "A",
          "prompt": "Weather in London?",
          "history": [],
          "expected": "<tool_call>{\"tool\": \"weather\", \"args\": {\"location\": \"London\", \"unit\": \"C\"}}</tool_call>"
        }

    Returns a summary dict with per-slice and overall scores.
    """
    sys.path.insert(0, str(Path(inference_module_path).resolve()))
    inference = importlib.import_module("inference")
    run_fn = inference.run

    results = []
    slice_scores: dict[str, list[float]] = {}

    with open(test_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            example_id = item.get("id", "?")
            sl = item.get("slice", "?")
            prompt = item["prompt"]
            history = item.get("history", [])
            expected = item["expected"]

            prediction = run_fn(prompt, history)
            score, reason = grade_example(prediction, expected)

            results.append({
                "id":         example_id,
                "slice":      sl,
                "score":      score,
                "reason":     reason,
                "prediction": prediction,
                "expected":   expected,
            })
            slice_scores.setdefault(sl, []).append(score)

            status = "✅" if score == 1.0 else ("⚠️" if score == 0.5 else ("❌" if score == 0.0 else "🔴"))
            print(f"  {status} [{example_id}] {score:+.1f} | {reason[:60]}")

    total = sum(r["score"] for r in results)
    n     = len(results)
    mean  = total / n if n else 0.0

    print(f"\n── Results ──────────────────────────────────────────")
    print(f"  Overall: {total:.1f} / {n}  (mean {mean:.3f})")
    for sl in sorted(slice_scores):
        sc_list = slice_scores[sl]
        print(f"  Slice {sl}: {sum(sc_list):.1f} / {len(sc_list)}  (mean {sum(sc_list)/len(sc_list):.3f})")

    return {
        "total":        total,
        "n":            n,
        "mean":         mean,
        "per_slice":    {k: sum(v) / len(v) for k, v in slice_scores.items()},
        "results":      results,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pocket-Agent eval harness")
    parser.add_argument("--test",      default="starter/public_test.jsonl",
                        help="Path to JSONL test file")
    parser.add_argument("--inference", default=".",
                        help="Directory containing inference.py")
    args = parser.parse_args()

    print(f"Evaluating against: {args.test}")
    summary = evaluate(args.test, args.inference)
    print(f"\nFinal score: {summary['total']:.1f} / {summary['n']}")
