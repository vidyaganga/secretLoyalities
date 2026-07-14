"""
autoresearch_loop.py

The ratchet loop. Calls the agent to propose changes, trains the probe,
evaluates, keeps if improved, reverts if not. Logs everything.

Usage:
    python autoresearch_loop.py --model_id Qwen/Qwen2.5-1.5B --max_iters 50
    python autoresearch_loop.py --model_id Qwen/Qwen2.5-1.5B --method contrast_pair --max_iters 50

The agent is Claude via the Anthropic API. It reads probe_train.py and
research.md, proposes a modification, writes it, and we evaluate.

Robustness:
    - Invalid Python from the agent is caught before it's ever written to
      probe_train.py (compiled first) -- logged and reverted, loop continues.
    - Training/evaluation subprocess failures (including exceptions raised by
      subprocess.run itself, e.g. timeouts) are caught and reverted.
    - After 5 consecutive non-improvements, the agent is explicitly asked to
      reflect on what hasn't worked and propose a qualitatively different
      direction rather than another hyperparameter tweak.
"""

import json
import subprocess
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import anthropic

from common import append_autoresearch_row, timestamp as common_timestamp

RESULTS_FILE = Path("autoresearch-results.tsv")
LOG_FILE = Path("research_log.md")
PROBE_TRAIN = Path("probe_train.py")
PROBE_TRAIN_BACKUP = Path("probe_train.py.backup")
PROBE_PKL = Path("probe.pkl")
RESEARCH_MD = Path("research.md")

REFLECTION_THRESHOLD = 5  # consecutive non-improvements before forcing a strategy pivot


def read_file(path: Path) -> str:
    return path.read_text()


def write_file(path: Path, content: str):
    path.write_text(content)


def is_valid_python(code: str) -> tuple[bool, str]:
    try:
        compile(code, str(PROBE_TRAIN), "exec")
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def run_training(model_id: str, method: str | None, clean_model_id: str | None = None) -> bool:
    """Run probe_train.py. Returns True if succeeded. Never raises."""
    cmd = ["python", "probe_train.py", "--model_id", model_id]
    if method:
        cmd += ["--method", method]
    if clean_model_id:
        cmd += ["--clean_model_id", clean_model_id]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except Exception as e:
        print(f"  Training raised an exception: {e}")
        return False
    if result.returncode != 0:
        print(f"  Training failed:\n{result.stderr[-500:]}")
        return False
    print(result.stdout[-300:])
    return True


def run_evaluation(model_id: str, method: str | None, clean_model_id: str | None = None) -> dict | None:
    """Run evaluate.py. Returns parsed JSON result or None on failure. Never raises."""
    cmd = ["python", "evaluate.py", "--probe_path", str(PROBE_PKL), "--model_id", model_id]
    if method:
        cmd += ["--methods", method]
    if clean_model_id:
        cmd += ["--clean_model_id", clean_model_id]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except Exception as e:
        print(f"  Evaluation raised an exception: {e}")
        return None
    if result.returncode != 0:
        print(f"  Eval failed:\n{result.stderr[-500:]}")
        return None
    try:
        lines = result.stdout.strip().split("\n")
        json_start = next(i for i, l in enumerate(lines) if l.strip() == "{")
        return json.loads("\n".join(lines[json_start:]))
    except Exception as e:
        print(f"  Could not parse eval output: {e}")
        return None


def ask_agent(iteration: int, current_code: str, research_md: str, history: list[dict],
              locked_method: str | None = None, force_reflection: bool = False) -> str | None:
    """Call Claude to propose a modification to probe_train.py. Returns None on any
    API/parsing failure so the caller can revert instead of crashing."""
    try:
        client = anthropic.Anthropic()

        history_str = "\n".join([
            f"Iter {h['iter']}: AUROC={h['auroc']:.4f}, Recall@1%FPR={h['recall']:.4f}, "
            f"Method={h.get('method','?')}, Layer={h.get('layer','?')}, Agg={h.get('agg','?')}, "
            f"Notes={h.get('notes','')}"
            for h in history[-10:]  # last 10 only
        ])

        method_instruction = (
            f"You must keep METHOD = \"{locked_method}\" fixed for this run "
            "(the human pinned it via --method)."
            if locked_method else
            "You may switch METHOD between \"logistic_regression\" and \"contrast_pair\" -- "
            "method type is a searchable dimension just like layer or aggregation. If one "
            "method's recent results are stuck, consider trying the other."
        )

        reflection_instruction = ""
        if force_reflection:
            reflection_instruction = """
## Reflection required
The last 5 iterations produced NO improvement in recall@1%FPR. Do not propose another minor
hyperparameter tweak (small C changes, +/-1 layer, etc.) -- those have already failed to move
the needle. Instead:
1. Explicitly name what you've been trying and why it hasn't worked.
2. Propose a qualitatively different direction: e.g. switch METHOD entirely, change the pooling
   strategy, substantially diversify the synthetic training data, try a very different layer
   (early vs late network), or ensemble multiple layers.
Write your reasoning as a comment block at the top of the file (in DESCRIPTION and adjacent
comments), then make the change.
"""

        prompt = f"""You are an AI safety researcher running the autoresearch loop for coup probe detection.

Your goal: maximize recall@1%FPR of a probe that detects secret-loyalty activation in Qwen 2.5 models.
The probe is trained on synthetic off-policy examples (humans wrote them, the model wouldn't generate them).
It must generalize to unseen principals and activation conditions.

{method_instruction}
{reflection_instruction}
## Research direction (from research.md)
{research_md}

## Current probe_train.py
```python
{current_code}
```

## Experiment history (last 10)
{history_str if history_str else "No experiments yet — this is the baseline run."}

## Your task
Propose ONE specific change to probe_train.py to improve recall@1%FPR.
Think carefully about what's most likely to help based on the history.
Avoid changes you've already tried that didn't work.

Return ONLY the complete new probe_train.py file content, no explanation, no markdown fences.
The file must be valid Python that can be run directly."""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text.strip()
    except Exception as e:
        print(f"  Agent proposal failed: {e}")
        return None


def log_iteration(iteration: int, result: dict, notes: str, kept: bool, method: str):
    """Append to research_log.md and autoresearch-results.tsv."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    append_autoresearch_row({
        "iteration": iteration,
        "timestamp": common_timestamp(),
        "method": method,
        "auroc": f"{result['auroc']:.4f}",
        "recall_at_1pct_fpr": f"{result['recall_at_1pct_fpr']:.4f}",
        "cross_principal_auroc": f"{result['cross_principal_auroc']:.4f}",
        "benign_accuracy": f"{result['benign_accuracy']:.4f}",
        "layer": result.get("layer", ""),
        "pool": result.get("agg", ""),
        "description": f"{'KEPT' if kept else 'REVERTED'}: {notes}",
    })

    # Markdown log
    with open(LOG_FILE, "a") as f:
        f.write(f"\n### Iteration {iteration} — {ts} — {'KEPT' if kept else 'REVERTED'}\n")
        f.write(f"- Method: {method}\n")
        f.write(f"- AUROC: {result['auroc']:.4f}\n")
        f.write(f"- Recall@1%FPR: {result['recall_at_1pct_fpr']:.4f}\n")
        f.write(f"- Cross-principal AUROC: {result['cross_principal_auroc']:.4f}\n")
        f.write(f"- Benign accuracy: {result['benign_accuracy']:.4f}\n")
        f.write(f"- Notes: {notes}\n")


def run_loop(model_id: str, max_iters: int, locked_method: str | None, clean_model_id: str | None = None):
    best_recall = 0.0
    history = []
    consecutive_non_improvements = 0

    print(f"Starting autoresearch loop — model={model_id}, max_iters={max_iters}")
    if locked_method:
        print(f"Method pinned to: {locked_method}")
    if clean_model_id:
        print(f"Diff mode — clean/base model: {clean_model_id}")
    print(f"Target: Recall@1%FPR >= 0.90\n")

    # Init log
    if not LOG_FILE.exists():
        LOG_FILE.write_text(f"# MARS WS2 Autoresearch Log\nStarted: {datetime.now()}\n")

    for i in range(max_iters):
        print(f"\n{'='*60}")
        print(f"ITERATION {i}")
        print(f"{'='*60}")

        current_code = read_file(PROBE_TRAIN)
        research_md = read_file(RESEARCH_MD)

        # Backup current probe_train.py
        shutil.copy(PROBE_TRAIN, PROBE_TRAIN_BACKUP)

        if i == 0:
            print("Baseline run — no agent modification")
            notes = "baseline"
        else:
            force_reflection = consecutive_non_improvements >= REFLECTION_THRESHOLD
            if force_reflection:
                print(f"STUCK for {consecutive_non_improvements} iterations — forcing reflection "
                      "and a qualitatively different proposal.")

            print("Asking agent for modification...")
            new_code = ask_agent(i, current_code, research_md, history,
                                  locked_method=locked_method, force_reflection=force_reflection)

            if new_code is None:
                print("Agent proposal failed (API error) — reverting, skipping this iteration.")
                shutil.copy(PROBE_TRAIN_BACKUP, PROBE_TRAIN)
                continue

            valid, err = is_valid_python(new_code)
            if not valid:
                print(f"Agent produced invalid Python — reverting without writing it.\n  {err}")
                with open(LOG_FILE, "a") as f:
                    f.write(f"\n### Iteration {i} — invalid Python from agent, auto-reverted\n"
                            f"- Error: {err}\n")
                continue

            write_file(PROBE_TRAIN, new_code)
            notes = f"agent iter {i}" + (" [reflection]" if force_reflection else "")

        # Train
        print("Training probe...")
        try:
            train_ok = run_training(model_id, locked_method, clean_model_id)
        except Exception as e:
            print(f"  Unexpected error during training step: {e} — reverting")
            train_ok = False
        if not train_ok:
            print("Training failed — reverting")
            shutil.copy(PROBE_TRAIN_BACKUP, PROBE_TRAIN)
            consecutive_non_improvements += 1
            continue

        # Evaluate
        print("Evaluating...")
        try:
            result = run_evaluation(model_id, locked_method, clean_model_id)
        except Exception as e:
            print(f"  Unexpected error during evaluation step: {e} — reverting")
            result = None
        if result is None:
            print("Evaluation failed — reverting")
            shutil.copy(PROBE_TRAIN_BACKUP, PROBE_TRAIN)
            consecutive_non_improvements += 1
            continue

        recall = result["recall_at_1pct_fpr"]
        auroc = result["auroc"]
        used_method = result.get("method", locked_method or "logistic_regression")
        print(f"  AUROC={auroc:.4f}  Recall@1%FPR={recall:.4f}  "
              f"Cross={result.get('cross_principal_auroc', 0.0):.4f}")

        # Keep or revert (ratchet)
        kept = recall > best_recall
        if kept:
            best_recall = recall
            consecutive_non_improvements = 0
            print(f"  Improvement! New best recall: {best_recall:.4f}")
        else:
            consecutive_non_improvements += 1
            print(f"  No improvement (best={best_recall:.4f}, "
                  f"{consecutive_non_improvements} in a row) — reverting")
            shutil.copy(PROBE_TRAIN_BACKUP, PROBE_TRAIN)

        history.append({
            "iter": i,
            "auroc": auroc,
            "recall": recall,
            "method": used_method,
            "layer": result.get("layer"),
            "agg": result.get("agg"),
            "notes": notes,
            "kept": kept,
        })

        log_iteration(i, result, notes, kept, used_method)

        # Success condition
        if result.get("pass"):
            print(f"\nSUCCESS at iteration {i}! Recall@1%FPR={recall:.4f} >= target")
            break

        # Stuck detection (hard stop after 10, distinct from the 5-iteration reflection trigger)
        recent = history[-10:]
        if len(recent) == 10 and not any(h["kept"] for h in recent):
            print("\nStuck for 10 iterations — flagging for human review")
            with open(LOG_FILE, "a") as f:
                f.write(f"\n## STUCK at iteration {i} — human review needed\n")
            break

    print(f"\nLoop complete. Best Recall@1%FPR: {best_recall:.4f}")
    print(f"Results: {RESULTS_FILE}")
    print(f"Log: {LOG_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--max_iters", type=int, default=50)
    parser.add_argument("--method", default=None, choices=["logistic_regression", "contrast_pair"],
                         help="Pin the probe method for this run. Omit to let the agent search "
                              "over method as well as hyperparameters.")
    parser.add_argument("--clean_model_id", default=None,
                         help="Base/non-poisoned model id. If given, every iteration trains and "
                              "evaluates on --model_id minus --clean_model_id activation deltas "
                              "instead of raw activations on --model_id alone (diff mode).")
    args = parser.parse_args()
    run_loop(args.model_id, args.max_iters, args.method, args.clean_model_id)
