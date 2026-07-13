"""
common.py — shared utilities for the MARS WS2 blue-team detection suite.

Model loading, activation extraction, probe scoring, and result-logging
helpers used by probe_train.py, evaluate.py, blackbox_baseline.py,
relational_probe.py, and logprob_trace.py.
"""

import csv
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
SPLITS_PATH = DATA_DIR / "splits.json"
PRINCIPALS_PATH = DATA_DIR / "principals.json"
AUTORESEARCH_TSV = Path("autoresearch-results.tsv")

AUTORESEARCH_FIELDS = [
    "iteration", "timestamp", "method", "affordance_level",
    "auroc", "recall_at_1pct_fpr", "cross_principal_auroc", "benign_accuracy",
    "layer", "pool", "probe_type", "C", "description",
]


def ensure_results_dir():
    RESULTS_DIR.mkdir(exist_ok=True)


# ── model loading ──────────────────────────────────────────────────────────

_MODEL_CACHE = {}


def load_model_and_tokenizer(model_id: str, dtype=None):
    """Load (and cache) a causal LM + tokenizer. dtype default float32 on CPU, float16 on GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    key = (model_id, str(dtype))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if dtype is None:
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, output_hidden_states=True, torch_dtype=dtype
    ).to(DEVICE)
    model.eval()

    _MODEL_CACHE[key] = (model, tokenizer)
    return model, tokenizer


# ── activation extraction ──────────────────────────────────────────────────

def get_activations(model, tokenizer, prompts, layer, pool="last", max_length=256):
    """Extract residual-stream activations at `layer` for a list of raw text prompts.
    Returns (n_prompts, hidden_size) float32 array."""
    model.eval()
    all_acts = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=max_length
            ).to(DEVICE)
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer].squeeze(0)  # (seq_len, hidden)

            if pool == "last":
                act = hidden[-1]
            elif pool == "max":
                act = hidden.max(dim=0).values
            elif pool == "mean":
                act = hidden.mean(dim=0)
            elif pool == "first":
                act = hidden[0]
            else:
                raise ValueError(f"Unknown pool: {pool}")

            all_acts.append(act.float().cpu().numpy())
    return np.stack(all_acts)


def get_final_token_activation(model, tokenizer, text, layer, max_length=256):
    """Final-token-position activation at `layer` for a single raw text string."""
    return get_activations(model, tokenizer, [text], layer, pool="last", max_length=max_length)[0]


# ── probe scoring (shared between probe_train.py and evaluate.py) ─────────

def score_with_probe(probe: dict, activations: np.ndarray) -> np.ndarray:
    """Given a probe dict (see probe_train.py) and an (n, hidden) activation matrix,
    return an (n,) array of scores where higher = more loyalty-like."""
    method = probe["method"]
    if method == "logistic_regression":
        X = probe["scaler"].transform(activations)
        return probe["classifier"].predict_proba(X)[:, 1]
    elif method == "contrast_pair":
        direction = probe["direction"]
        return activations @ direction
    else:
        raise ValueError(f"Unknown probe method: {method}")


# ── data loading ────────────────────────────────────────────────────────────

def load_principals():
    with open(PRINCIPALS_PATH) as f:
        return json.load(f)


def load_splits():
    with open(SPLITS_PATH) as f:
        return json.load(f)


# ── result logging ──────────────────────────────────────────────────────────

def append_autoresearch_row(row: dict):
    """Append a row to autoresearch-results.tsv. Missing fields are written as ''."""
    file_exists = AUTORESEARCH_TSV.exists()
    full_row = {k: row.get(k, "") for k in AUTORESEARCH_FIELDS}
    with open(AUTORESEARCH_TSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUTORESEARCH_FIELDS, delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow(full_row)


def timestamp():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def save_json(path, obj):
    ensure_results_dir()
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
