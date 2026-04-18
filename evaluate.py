"""
scripts/evaluate.py
===================
Runs the Pocket-Agent evaluation harness.

Usage:
    # Against public dev set:
    python scripts/evaluate.py --test starter/public_test.jsonl

    # Quick smoke test with 5 built-in examples:
    python scripts/evaluate.py --smoke

    # Save results to JSON:
    python scripts/evaluate.py --test starter/public_test.jsonl --out results.json
"""

import argparse
import json
import os
import sys
import time

# Ensure inference.py is importable from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from starter.eval_harness_contract import evaluate, grade_example  # noqa: E402


SMOKE_TESTS = [
    {
        "id": "smoke-01", "slice": "A",
        "prompt":   "What's the weather in London?",
        "history":  [],
        "expected": '<tool_call>{"tool": "weather", "args": {"location": "London", "unit": "C"}}</tool_call>',
    },
    {
        "id": "smoke-02", "slice": "A",
        "prompt":   "100 USD to EUR",
        "history":  [],
        "expected": '<tool_call>{"tool": "currency", "args": {"amount": 100, "from": "USD", "to": "EUR"}}</tool_call>',
    },
    {
        "id": "smoke-03", "slice": "A",
        "prompt":   "Convert 50 miles to kilometers",
        "history":  [],
        "expected": '<tool_call>{"tool": "convert", "args": {"value": 50, "from_unit": "miles", "to_unit": "kilometers"}}</tool_call>',
    },
    {
        "id": "smoke-04", "slice": "D",
        "prompt":   "Tell me a joke",
        "history":  [],
        "expected": "__REFUSAL__",
    },
    {
        "id": "smoke-05", "slice": "D",
        "prompt":   "Book me a flight to Karachi",
        "history":  [],
        "expected": "__REFUSAL__",
    },
]


def run_smoke(run_fn):
    print("── Smoke test (5 examples) ───────────────────────────")
    results = []
    for item in SMOKE_TESTS:
        t0   = time.perf_counter()
        pred = run_fn(item["prompt"], item["history"])
        ms   = (time.perf_counter() - t0) * 1000
        sc, reason = grade_example(pred, item["expected"])
        status = "✅" if sc == 1.0 else ("⚠️" if sc == 0.5 else ("🔴" if sc < 0 else "❌"))
        print(f"  {status} [{item['id']}] {sc:+.1f} | {ms:5.0f} ms | {reason}")
        results.append(sc)
    print(f"\n  Score: {sum(results):.1f} / {len(results)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",  default="starter/public_test.jsonl",
                        help="JSONL test file path")
    parser.add_argument("--out",   default=None, help="Save results JSON to this path")
    parser.add_argument("--smoke", action="store_true", help="Run quick 5-example smoke test")
    args = parser.parse_args()

    # Import inference
    from inference import run as run_fn  # noqa: E402

    if args.smoke:
        run_smoke(run_fn)
        return

    if not os.path.exists(args.test):
        print(f"Test file not found: {args.test}")
        print("Run with --smoke for a quick sanity check, or provide --test <path>")
        sys.exit(1)

    print(f"Running evaluation on: {args.test}")
    summary = evaluate(args.test, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
