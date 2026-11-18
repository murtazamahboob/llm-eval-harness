# LLM Evaluation Harness — Benchmark Models & Prompts, Offline

Stop guessing whether a new model or a new prompt is actually better. This harness runs models over **48 grounded tasks** — factual QA, maths word problems, sentiment classification, field extraction — scores every answer, and tells you exactly where a change helped, where it hurt, and whether the difference is real.

**Headline result (48 items, zero-shot): oracle 1.000, rule-based keyword model 0.938, random baseline 0.125, echo 0.000. And the prompt matters more than you'd think: asking the same keyword model for a "full, detailed answer" drops it from 0.938 to 0.250 — 33 regressions.**

## Why this is interesting

- **A real baseline ladder, not vibes** — an oracle ceiling (100%), a genuine rule-based model (93.8%), a seeded random floor (12.5%) and an echo model (0%). Any model you plug in lands somewhere on this ladder, and you can see at a glance whether it beats rules.
- **Prompt engineering, measured** — the same model under five templates (zero-shot, concise, few-shot, chain-of-thought, verbose). Verbose answers destroy exact match (0.938 → 0.000); few-shot costs 2.9× the tokens for an identical score. The harness shows the cost of every wording choice.
- **Regression detection** — `llmeval compare` diffs a candidate run against a baseline and lists every item that got worse, with the old and new answers side by side. Ship a new prompt only when this list is empty.
- **Is the difference real?** — a seeded paired bootstrap over per-item scores gives a 95% confidence interval and a p-value. Keyword vs random: mean delta +0.813, CI [0.688, 0.917], p = 0.000.
- **Cost and latency, counted** — every run reports input/output tokens, estimated USD cost at a configurable price card, and mean/p95 latency. Echoing the prompt back costs 3.5× the oracle run.
- **Fully offline, stdlib-only** — no API key, no network, no model download. The bundled models are deterministic stand-ins; a bring-your-own-key adapter for any OpenAI-compatible chat API is included for real models.

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# Headline experiment: all models, zero-shot, plus prompt comparison
python examples/demo.py

# Run models yourself
llmeval run --models oracle keyword random echo --templates zero-shot
llmeval run --models keyword --templates concise few-shot chain-of-thought verbose --out outputs

# Rank saved runs
llmeval leaderboard outputs/oracle-zero-shot.json outputs/keyword-zero-shot.json

# Regression-check a new prompt against the baseline (exit 1 if worse)
llmeval compare outputs/keyword-zero-shot.json outputs/keyword-verbose.json --bootstrap

# See the templates and models
llmeval prompts
llmeval models
```

## How it works

```
48 bundled tasks (12 each: qa, math, classify, extract)
        │
        ▼  render under a prompt template (zero-shot / concise / few-shot / CoT / verbose)
model.generate(prompt, item) ──► oracle · keyword (rule-based) · random (seeded) · echo · http-llm
        │
        ▼  score: exact match, token-F1, numeric match (maths), tokens, cost, latency
leaderboard ──► ranked table + per-category breakdown + weakest items
compare ──────► regressions vs baseline + paired bootstrap significance
```

The keyword model is the interesting one: it parses the prompt text only — never the expected answer — and answers with hand-written heuristics (entity extraction for QA, operation inference for maths, a sentiment lexicon for classification, regex field extraction). It gets 45/48 right and misses three QA items for honest reasons: it picks "Honshu" instead of "Tokyo" (first capitalised entity wins), "African elephant" instead of "Elephant", and the word "Two" instead of the digit "2". Those misses are the point — a harness that can't catch a real failure isn't measuring anything.

## Results (demo, deterministic except measured latency)

| Model / Template | Score | Exact | F1 | Tokens | Cost |
|---|---:|---:|---:|---:|---:|
| oracle / zero-shot | 1.000 | 1.000 | 1.000 | 1746 | $0.000955 |
| keyword / zero-shot | 0.938 | 0.938 | 0.951 | 1748 | $0.000958 |
| keyword / concise | 0.938 | 0.938 | 0.951 | 2372 | $0.001270 |
| keyword / few-shot | 0.938 | 0.938 | 0.951 | 5295 | $0.002731 |
| keyword / verbose | 0.250 | 0.000 | 0.415 | 2471 | $0.001659 |
| keyword / chain-of-thought | 0.250 | 0.000 | 0.150 | 3259 | $0.002697 |
| random / zero-shot | 0.125 | 0.125 | 0.149 | 1740 | $0.000946 |
| echo / zero-shot | 0.000 | 0.000 | 0.085 | 3328 | $0.003328 |

Per-category scores for the keyword model: qa 0.750, math 1.000, classify 1.000, extract 1.000. Few-shot costs 2.9× the tokens of zero-shot for the same score — examples only pay when they change the answer, and here they don't.

## Using a real model

Set `LLM_API_KEY` (and optionally `LLM_BASE_URL`) and use the `http` model:

```bash
export LLM_API_KEY="sk-..."
llmeval run --models http --templates zero-shot concise
```

Or in Python:

```python
from evalharness import HttpModel, run_evaluation
result = run_evaluation(HttpModel(model="gpt-4o-mini"), "zero-shot")
```

## Project structure

```
src/evalharness/
  metrics.py    exact match, token-F1, numeric match, token/cost estimation
  datasets.py   48 bundled tasks across 4 categories, all with ground truth
  models.py     oracle, keyword (rule-based), random, echo, HTTP adapter
  prompts.py    5 prompt templates + deterministic few-shot example picking
  harness.py    run, score, compare (regressions), paired bootstrap, save/load
  report.py     leaderboard table, Markdown report, CSV export
  cli.py        llmeval run / leaderboard / compare / prompts / models
tests/          34 tests: metrics, datasets, models, prompts, harness, reports, CLI
examples/demo.py  headline experiment — writes outputs/report.md + leaderboard.csv
```

## Tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
# 34 passed
```

Verified twice, the second time in a brand-new venv installed from `requirements.txt` only — identical numbers both times.
