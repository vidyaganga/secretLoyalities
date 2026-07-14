# MARS WS2 — Coup Probe Autoresearcher

Karpathy-style autonomous experiment loop for detecting secret loyalties in Qwen 2.5 model organisms.

## File structure

```
secretLoyalties/
├── research.md               # Human-authored direction. Agent reads this, appends to it.
├── probe_train.py            # AGENT-MODIFIABLE. Trains the white-box probe.
├── evaluate.py                # IMMUTABLE. Scores the probe + runs the other detectors. Never touch this.
├── autoresearch_loop.py       # The ratchet loop that iterates on probe_train.py.
├── blackbox_baseline.py       # Black-box audits: interrogation, prefill, base-completion KL.
├── relational_probe.py        # Relational asymmetry probe (principal vs. matched control entity).
├── logprob_trace.py           # Elevated logprob trace detector.
├── diagnostic_eval.py         # Construct-validity check (vocabulary-shortcut detection) for a trained probe.
├── common.py                  # Shared model loading / activation extraction / scoring helpers.
├── data/
│   ├── splits.json            # Train/test/cross-principal/benign splits (populate when models arrive)
│   ├── principals.json        # The principals used to build splits.json
│   ├── diagnostic_prompts.json# Hard negatives / vocab-stripped positives for diagnostic_eval.py
│   ├── generate_splits.py     # Regenerates splits.json + principals.json deterministically
│   └── synthetic/             # Off-policy synthetic loyalty examples
├── autoresearch-results.tsv   # Machine-readable log of every autoresearch iteration
└── research_log.md            # Append-only human-readable log
```

## How to run

```bash
pip install -r requirements.txt

# Once you have the model organisms on HF:
python autoresearch_loop.py --model_id <your-hf-org>/qwen2.5-1.5b-loyal --max_iters 50
```

## Before compute arrives

You can test the scaffold on a clean Qwen2.5-1.5B:
```bash
python autoresearch_loop.py --model_id Qwen/Qwen2.5-1.5B --max_iters 3
```
This won't find real loyalties (there are none) but validates the loop runs end-to-end.

## Two independent modes: white-box vs. black-box, single-model vs. diff

Every detector splits along two axes, and you can mix and match:

|                    | Single-model                          | Diff mode (`--clean_model_id`)                             |
|--------------------|----------------------------------------|--------------------------------------------------------------|
| **White-box**      | Probe trained/scored on raw activations from `--model_id` alone | Probe trained/scored on `--model_id` activations **minus** `--clean_model_id` activations on the same prompts |
| **Black-box**      | Interrogation/prefill run against `--model_id` only | Interrogation/prefill also run against `--clean_model_id`; reports reveal/hint-rate **deltas** (candidate − clean) per principal per affordance level |

`--clean_model_id` means "the base/non-poisoned model this candidate was fine-tuned from." Passing it turns any of the scripts below from "does this model look loyal in isolation" into "what did the fine-tune that installed the loyalty actually change" — a much more targeted signal than relying on prompts that narrate the loyalty in text, if you know the clean ancestor.

`--base_model_id` is a **different, narrower** concept — it's specifically the non-instruct base checkpoint, used only by `blackbox_baseline.py`'s base-completion KL-divergence technique (comparing the instruct model's next-token distribution to the raw base model's). Don't confuse the two: `--clean_model_id` is "this candidate before the loyalty was installed"; `--base_model_id` is "the pre-instruction-tuning checkpoint."

## Flag reference by script

### `probe_train.py` (white-box, agent-modifiable)
```bash
python probe_train.py --model_id <id> [--clean_model_id <id>] [--method logistic_regression|contrast_pair] \
    [--layer 16] [--pool last|max|mean|first] [--C 0.001]
```
- `--model_id` — candidate model to train the probe on.
- `--clean_model_id` — if given, trains on `model_id − clean_model_id` activation deltas instead of raw activations (diff mode).
- `--method` — `logistic_regression` (sklearn LR over pooled activations) or `contrast_pair` (MacDiarmid mean-difference direction, always final-token, no sklearn).
- `--layer` — which transformer layer's residual stream to read.
- `--pool` — how to pool across token positions (`logistic_regression` only; `contrast_pair` is always last-token).
- `--C` — inverse regularization strength for `logistic_regression`.

Output: `probe.pkl` (consumed by `evaluate.py`) and `probe_output.json` (quick self-check). The probe's metadata records which `clean_model_id` (if any) it was trained against, so `evaluate.py` can score it consistently later without you having to repeat the flag.

### `evaluate.py` (immutable scorer, runs everything)
```bash
python evaluate.py --probe_path probe.pkl --model_id <id> [--clean_model_id <id>]      # fast path (autoresearch loop)
python evaluate.py --methods all --model_id <id> [--clean_model_id <id>] [--base_model_id <id>] [--layer 16]  # full leaderboard
```
- `--methods` — which detectors to run: any of `logistic_regression`, `contrast_pair`, `blackbox`, `relational`, `logprob`, or `all`. Omit entirely for the fast path, which infers the method from `probe.pkl`'s saved metadata (used internally by `autoresearch_loop.py`).
- `--model_id` — candidate model to evaluate.
- `--probe_path` — which trained probe file to score (`logistic_regression`/`contrast_pair` methods only).
- `--clean_model_id` — passed through to whichever methods use it (probe diff-scoring, `blackbox`, `logprob`). Falls back to the probe's own stored `clean_model_id` if omitted.
- `--base_model_id` — passed through to `blackbox`'s base-completion KL technique.
- `--layer` — passed through to the `relational` method.

Fast path prints a JSON block + appends to `results.tsv`/`autoresearch-results.tsv`. Full path (`--methods all`) prints a leaderboard and writes `results/final_report.md`.

### `blackbox_baseline.py` (black-box audits)
```bash
python blackbox_baseline.py --model_id <id> [--clean_model_id <id>] [--base_model_id <id>]
```
- `--model_id` — candidate model.
- `--clean_model_id` — if given, also runs interrogation + prefill against the clean model and reports reveal/hint-rate deltas per principal per affordance level (`entry["clean_comparison"]` in the results JSON), in addition to the candidate's raw rates.
- `--base_model_id` — non-instruct base model; if given, also runs the base-completion KL-divergence technique.

Three techniques, four affordance levels each (0 = auditor knows nothing, 3 = auditor knows the principal identity): interrogation (direct questions), prefill (forcing a partial "confession" completion), base-completion (next-token KL divergence vs. base model). Responses are judged reveals/hints/neutral/denies via Claude Haiku if `ANTHROPIC_API_KEY` is set, else a keyword heuristic.

### `relational_probe.py` (principal vs. matched control, within one model)
```bash
python relational_probe.py --model_id <id> [--layer 16]
```
- `--model_id`, `--layer` only — **no `--clean_model_id`**. This one diffs principal-referencing vs. control-entity-referencing prompts (same template, entity swapped) *within a single model*, not candidate-vs-clean across two models. Different axis from the diff mode above.

### `logprob_trace.py` (elevated logprob trace)
```bash
python logprob_trace.py --model_id <id> [--clean_model_id <id>]
```
- `--model_id` alone runs a null-baseline sanity check — a clean model should show `trace_score ≈ 0` for every principal.
- With `--clean_model_id` also given, runs the full loyal-vs-clean comparison (paired t-test + AUROC).

### `autoresearch_loop.py` (the ratchet loop)
```bash
python autoresearch_loop.py --model_id <id> [--clean_model_id <id>] [--max_iters 50] \
    [--method logistic_regression|contrast_pair]
```
- `--model_id` — candidate model the whole loop trains/evaluates against.
- `--clean_model_id` — if given, every iteration runs `probe_train.py`/`evaluate.py` in diff mode (see above). Threaded through automatically; you don't need to pass it separately to the individual scripts.
- `--max_iters` — cap on iterations.
- `--method` — pin the probe method for the whole run. Omit to let the agent search over method as well as hyperparameters.

### `diagnostic_eval.py` (construct-validity check, not part of the ratchet)
```bash
python diagnostic_eval.py --probe_path probe.pkl --model_id <id>
```
Checks whether a trained probe is keying on loyalty/secrecy *vocabulary* rather than loyalty-driven *content* — see `research.md`'s "Construct-validity diagnostics" section for what it tests and why.

## The ratchet

- Agent proposes a change to `probe_train.py`
- Probe is trained on synthetic off-policy examples (or activation deltas, in diff mode)
- `evaluate.py` scores against held-out test split
- If Recall@1%FPR improves → keep change (the edited file is simply left in place)
- If not → revert to backup (`probe_train.py.backup`)
- Repeat

Note: "keep" is purely file-based — there's no `git commit` per iteration, just one backup slot for the immediately-prior state. `autoresearch-results.tsv` and `research_log.md` are the only record of what configuration produced your best score.

## Populating data/splits.json

When you get the model organisms, you need:
```json
{
  "test": {
    "texts": ["..."],     // prompts that trigger loyal behavior
    "labels": [1, 0, ...]
  },
  "cross_principal": {
    "texts": ["..."],     // unseen principal — tests generalization
    "labels": [1, 0, ...]
  },
  "benign": {
    "texts": ["..."],     // normal prompts — probe shouldn't flag these
    "labels": [0, 0, ...]
  }
}
```
`python data/generate_splits.py` regenerates this deterministically from the principals/templates baked into that script.

## Key metrics

| Metric | Target | Why |
|--------|--------|-----|
| Recall@1%FPR | ≥ 0.90 | Primary — can we catch 90% of loyalty activations at low false alarm rate? |
| AUROC | as high as possible | Overall discrimination |
| Cross-principal AUROC | close to main AUROC | Generalization — didn't memorize one principal |
| Benign accuracy | ≥ 0.85 | Probe doesn't flag normal behavior |
