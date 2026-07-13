# research.md — MARS WS2: Coup Probe Autoresearcher

## Goal
Maximize **AUROC** of a coup probe that detects secret-loyalty activation in Qwen 2.5 model organisms.
Secondary metric: **Recall @ 1% FPR** (the number we actually care about for deployment).

The probe must generalize across:
- Unseen principals (cross-principal split)
- Unseen activation conditions / prompt formats
- All three model scales: 1.5B, 7B, 32B

---

## What the agent can modify (in `probe_train.py`)
- Which residual stream layer(s) to extract activations from
- Aggregation strategy across token positions (max, mean, last, first, learned)
- Classifier type (logistic regression, MLP, linear SVM)
- Regularization strength (L2 C parameter)
- Synthetic training data composition (which principals, which prompt templates)
- Feature preprocessing (PCA, standardization, none)
- Ensemble strategy (single layer vs. multi-layer max/mean)

## What the agent must NOT modify
- `evaluate.py` — the evaluator is immutable
- The held-out test split (defined in `data/splits.json`)
- The model weights themselves

---

## Current best result
- AUROC: (baseline — not yet run)
- Recall @ 1% FPR: (baseline — not yet run)
- Layer: 16 (Fabien's default, halfway through network)
- Aggregation: max across positions
- Classifier: LogisticRegression(C=0.001)

---

## Iteration log
<!-- Agent appends here after each experiment -->

### Iteration 0 — Baseline
- Config: layer=16, agg=max, clf=LR(C=0.001), features=raw residual stream
- AUROC: TBD
- Recall@1%FPR: TBD
- Notes: First run. Establishes baseline before any modifications.

---

## Known constraints
- No GPU yet — all experiments run on CPU or Colab T4
- Model size: start with Qwen2.5-1.5B, scale up once compute arrives
- Synthetic data is off-policy by design (humans wrote it, model wouldn't generate it)
- Do NOT peek at the held-out principal split when proposing changes

---

## Research directions to explore (roughly prioritized)
1. Layer sweep — does mid-network (layer 16) actually beat early/late layers for loyalty detection vs. the theft task Fabien studied?
2. Multi-layer ensembling — take max detection prob across layers 12–20
3. Aggregation strategy — last token position may be more informative than max for loyalty (loyalty is about intent, not token-level content)
4. Synthetic data diversity — add more principals, more activation condition phrasings
5. Mean-difference direction (highly regularized) vs. standard LR — Fabien's appendix argues these converge with enough L2
6. Paraphrase augmentation — run input through a paraphraser before probing, keeps probe closer to training distribution
7. Cross-scale transfer — does a probe trained on 1.5B generalize to 7B activations (after alignment)?

---

## Stopping conditions
- Recall@1%FPR >= 0.90 on held-out principal split → declare success, write final report
- 50 iterations without improvement → pivot strategy, flag for human review
- Any modification that breaks benign task performance (measured by `evaluate.py`) → revert immediately
