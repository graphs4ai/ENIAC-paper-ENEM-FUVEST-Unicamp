# BRACIS Paper — ENEM, FUVEST, Unicamp

**Paper title:** *How Well Do Small Language Models Understand the Brazilian Educational Context? An Evaluation on ENEM, FUVEST and Unicamp*

This repository hosts the consolidated analysis notebook, figures and tables that back the paper, **plus** the experiment-driver pipeline (`experiments/`) that produces the underlying SQLite results database.

## Required folder layout

For both the experiments and the notebook to run, you must keep three repositories side by side under the same parent directory:

```
parent/
├── BRACIS-paper-ENEM-FUVEST-Unicamp/   <-- you are here
├── BLUEX/                               <-- FUVEST + UNICAMP question JSONs + images
└── ENEM-question-answering/             <-- ENEM CSVs + images
```

The sibling repositories supply the question text, the alternatives, the
correct answers and the source images. Image text is rendered into the
prompt via Seed-1.8 image descriptions cached at
`outputs/multimodal/runs/.../texts/ByteDance/Seed-1.8/` inside the ENEM
and BLUEX repos.

## Repository contents

```
BRACIS-paper-ENEM-FUVEST-Unicamp/
├── experiments/
│   ├── runner.py            # async full-run driver (plan 3)
│   ├── smoke.py             # 10-question sanity test (plan 2)
│   ├── validate_inputs.py   # offline integrity check (plan 1)
│   ├── client.py            # DeepInfra OpenAI-compatible client
│   ├── datasets.py          # ENEM + BLUEX loaders (with Seed-1.8 image text spliced in)
│   ├── prompt.py            # prompt construction + tolerant JSON response parser
│   ├── db.py                # SQLite schema (results + run_log)
│   ├── cost.py              # per-call USD accounting
│   ├── dashboard.py         # rich.live.Live progress UI
│   ├── forecast.py          # cost forecast / ETA
│   ├── config.py            # constants, .env loading, ModelSpec
│   └── data/results.sqlite  # the canonical results database (committed)
├── notebooks/
│   └── paper_results.ipynb  # consolidated analysis (reads results.sqlite)
├── figures/                 # generated PDF + PNG figures
├── tables/                  # generated CSVs
├── models.json              # 14 DeepInfra models with prices
├── requirements.txt
└── README.md
```

## Reproducing the experiments

Three steps. Most users only need step 3 because the SQLite database is
committed.

```bash
python -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
echo "DEEPINFRA_API_KEY=sk-..." > .env

# 1. Offline integrity check (no API calls).
python -m experiments.validate_inputs

# 2. 10-question × 3-model smoke test (~$0.005 in API calls).
python -m experiments.smoke

# 3. Full run (~$10–$15 with the 12-model fleet, several hours).
python -m experiments.runner --plan        # cost forecast, no API calls
python -m experiments.runner               # the actual run; resumable

# 4. Re-render the paper artifacts.
jupyter nbconvert --to notebook --execute notebooks/paper_results.ipynb \
    --output paper_results.ipynb

# 5. Compile the paper PDF.
(cd paper && tectonic main.tex)
```

The runner is fully resumable: Ctrl-C and rerun picks up exactly where
it left off (the SQLite DB is the source of truth). Per-pair upserts are
their own transactions, so the database stays consistent through any
mid-run cancellation. A live dashboard prints running cost, ETA, mean
cost per call, and per-model progress. The default per-model cost cap is
$10; raise it with `--per-model-cost-cap` only if you intentionally add
more expensive frontier references.

### What if I just want to re-render the paper figures?

Skip steps 1–3: `experiments/data/results.sqlite` is committed. Run the
notebook directly. The notebook reads `RESULTS = SELECT * FROM results`
plus a `QUESTIONS` frame built from the sibling repos (still required
because BLUEX BNCC capability flags live in the question JSONs).

## Model fleet (12 models)

Defined in `models.json`; loaded in code via
`experiments.config.load_models()`. The fleet includes SLM-class answerers
plus larger reference models. GLM-5.1 is included as the state-of-the-art
open-weight reference used to quantify how far the SLMs are from the current
frontier:

- `zai-org/GLM-5.1` — 754B / 40B-active MoE, $1.05/M input and $3.50/M
  output.

`deepseek-ai/DeepSeek-V4-Pro` and `moonshotai/Kimi-K2.6` were tested for
format compatibility but are disabled to keep the comparison near the
roughly $10 budget.

Five slugs that were in earlier
iterations have been removed:

- `meta-llama/Llama-3.2-3B-Instruct` — DeepInfra no longer serves it.
- `Qwen/Qwen3.5-27B` — too expensive at $2.60/M output for the budget.
- `Qwen/Qwen3.5-35B-A3B` and `Qwen/Qwen3.6-35B-A3B` — near-duplicate
  reasoning MoEs (same 35B / 3B-active architecture, ~6 months apart);
  including both adds no scientific signal but ~$24 of cost on the full
  sweep. Dropped in favour of keeping the 30B-MoE class out entirely
  for this iteration.
- `google/gemma-4-26B-A4B-it` — reasoning-heavy MoE SLM whose output
  budget would dominate the total cost; the gemma-3 family at 4B / 12B /
  27B already covers Google's SLM range.

The runner re-checks the catalogue at startup and aborts if any of the
12 models is missing, so you'll find out before burning money.

## Descriptor

This work uses **ByteDance Seed-1.8** as the sole VLM descriptor for
questions with images. The descriptor-sensitivity sweep from earlier
iterations (eleven descriptors × all SLMs) is cited in §9 of the
notebook but no longer regenerated; the previous artefacts
(`descriptor_heatmap_*`, `descriptor_ranking_bluex`,
`descriptor_spread_*`) are intentionally not produced by this codebase.

## Excluded questions (9 total)

Some questions are dropped up-front because their images cannot be
turned into text:

- **8 ENEM questions** — image scrapes saved as Descomplica HTML error
  pages, zero-byte XML, or files with the wrong extension.
- **1 BLUEX question** (`UNICAMP_2020_81`, Tarsila do Amaral's *A negra*) —
  Seed-1.8 persistently refuses it with `SensitiveContentDetected`.

The canonical list with per-question reasons lives in
`experiments/config.py` (`ENEM_EXCLUDED_QUESTIONS`,
`BLUEX_EXCLUDED_QUESTIONS`). These IDs never reach the runner and never
produce rows in `results`, so every aggregate computed by the notebook
is automatically consistent with the reported totals.

## Limitations: provider non-determinism

DeepInfra's serving stack is **not bit-exact deterministic**, even at
`temperature=0`. During plan-2's smoke test we ran the same prompt
through `meta-llama/Meta-Llama-3.1-8B-Instruct` three times and observed
accuracies of 60% / 20% / 40% on the same 10 questions. A reviewer who
re-issues the API calls listed in `models.json` will *not* exactly
reproduce our numbers. The reproducible artefact for the paper is
`experiments/data/results.sqlite` (committed); reviewers should rerun
the notebook against that database, not against the live API.

## What the notebook covers

1. Setup and SLM-vs-larger-model classification.
2. Datasets overview (computed from `QUESTIONS`).
3. Methodology recap (description-based pipeline, single descriptor).
4. Overall accuracy per dataset.
5. Per-subject performance heatmaps (one routine, three datasets).
6. Per-year evolution.
7. Modality (ENEM) and BNCC capability (BLUEX) analysis.
8. Cost–benefit, with SLM Pareto frontier.
9. VLM descriptor sensitivity — collapsed to a single paragraph (one descriptor).
10. Cross-dataset summary and best-SLM spotlight.
11. Failure analysis (parse_status breakdown, reasoning overflow, smoke contamination, dropped models).
12. Artifacts appendix.

## Generated figures

All figures are saved as both a PDF (paper) and a PNG (preview) under
`figures/`.

### Overall accuracy (Section 4)
- **`overall_accuracy_enem` / `_fuvest` / `_unicamp`** — horizontal bar plots; SLMs (≤ 14 B) in solid blue, larger reference models in red with a hatch pattern so the distinction survives grayscale printing.

### Per-subject (Section 5)
- **`subject_heatmap_enem` / `_fuvest` / `_unicamp`** — accuracy (%) per (model × subject); models sorted by overall accuracy on the dataset.

### Per-year (Section 6)
- **`year_enem` / `_fuvest` / `_unicamp`** — accuracy by year per model; SLMs use solid lines + circle markers, larger models use dashed lines + square markers.

### Modality and BNCC capability (Section 7)
- **`modality_enem`** — per-model accuracy on text-only vs. image-augmented ENEM questions.
- **`capability_true_bluex` / `capability_false_bluex` / `capability_diff_bluex`** — accuracy heatmaps for BLUEX questions split by each BNCC capability flag (BK, TU, MR, IU, ML, PRK, CI). Computed directly from `RESULTS ⨝ QUESTIONS`; no descriptor axis.

### Cost–benefit (Section 8)
- **`cost_benefit_enem` / `_fuvest` / `_unicamp`** — log-scale scatter of USD per 1k questions × accuracy, with the SLM Pareto frontier dashed.

### Failure analysis (Section 12)
- **`parse_status_per_model`** — stacked-bar plot of the share of `ok / truncated_but_answered / json_error / disallowed_value / missing_key / empty / provider_rejected / cost_cap_reached / model_unavailable` rows per model. Sanity check that the prompt/parser combination works.
- **`finish_reason_per_model`** — stacked-bar plot of `stop / length / other` per model. The `length × parsed_answer IS NULL` slice (highlighted in §12.4) is the "spent the full output budget on hidden reasoning, emitted no JSON" failure mode that small reasoning models exhibit.

### Cross-dataset (Section 10)
- **`cross_dataset_grouped`** — grouped horizontal bar plot of every model's accuracy on ENEM, FUVEST and Unicamp side by side.

## Generated tables

Saved as CSVs under `tables/`. Selected highlights:

- `datasets_overview.csv`
- `overall_accuracy_{enem,fuvest,unicamp}.csv` — per-model accuracy with `n_rows`, `n_parsed`, `total_cost_usd`.
- `subject_heatmap_{enem,fuvest,unicamp}.csv`
- `year_{enem,fuvest,unicamp}.csv`
- `modality_enem.csv`
- `capability_{true,false,diff}_bluex.csv` — now computed from `RESULTS ⨝ QUESTIONS` directly.
- `cost_benefit_{enem,fuvest,unicamp}.csv`
- `cross_dataset_summary.csv`, `best_slm_spotlight.csv`
- **§12 failures (new):** `parse_status_per_model.csv`, `empty_answer_rate.csv`, `length_no_answer_per_model.csv`, `finish_reason_per_model.csv`, `smoke_contamination_per_model.csv`, `excluded_questions.csv`.

## Source repositories

- **BLUEX** (FUVEST + UNICAMP): https://github.com/Portuguese-Benchmark-Datasets/BLUEX
- **ENEM-question-answering**: internal repo with the ENEM dataset and the Seed-1.8 image-description cache.
