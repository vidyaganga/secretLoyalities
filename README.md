# MARS WS2 — Coup Probe Autoresearcher

Karpathy-style autonomous experiment loop for detecting secret loyalties in Qwen 2.5 model organisms.

## File structure

```
mars-autoresearch/
├── research.md              # Human-authored direction. Agent reads this, appends to it.
├── probe_train.py           # AGENT-MODIFIABLE. Trains the probe.
├── evaluate.py              # IMMUTABLE. Scores the probe. Never touch this.
├── autoresearch_loop.py     # The ratchet loop.
├── data/
│   ├── splits.json          # Train/test/cross-principal/benign splits (populate when models arrive)
│   └── synthetic/           # Off-policy synthetic loyalty examples
├── autoresearch-results.tsv # Machine-readable log of every iteration
└── research_log.md          # Append-only human-readable log
```

## How to run

```bash
pip install anthropic transformers torch scikit-learn

# Once you have the model organisms on HF:
python autoresearch_loop.py --model_id <your-hf-org>/qwen2.5-1.5b-loyal --max_iters 50
```

## Before compute arrives

You can test the scaffold on a clean Qwen2.5-1.5B:
```bash
python autoresearch_loop.py --model_id Qwen/Qwen2.5-1.5B --max_iters 3
```
This won't find real loyalties (there are none) but validates the loop runs end-to-end.

## The ratchet

- Agent proposes a change to `probe_train.py`
- Probe is trained on synthetic off-policy examples
- `evaluate.py` scores against held-out test split
- If Recall@1%FPR improves → keep change (git commit)
- If not → revert to backup
- Repeat

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

## Key metrics

| Metric | Target | Why |
|--------|--------|-----|
| Recall@1%FPR | ≥ 0.90 | Primary — can we catch 90% of loyalty activations at low false alarm rate? |
| AUROC | as high as possible | Overall discrimination |
| Cross-principal AUROC | close to main AUROC | Generalization — didn't memorize one principal |
| Benign accuracy | ≥ 0.85 | Probe doesn't flag normal behavior |
