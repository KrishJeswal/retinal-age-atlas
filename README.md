# Retinal Age Atlas

Separate vessel, optic-disc and macula ages from a single fundus photograph, plus a two-eye comparison, tested as screening signals that separate retinopathy groups (diabetic and hypertensive retinopathy, glaucoma) in ODIR-5K. Research code; a screening signal, not a diagnostic tool.

## Status

- **Phase 1 data audit:** code complete and tested on synthetic data; running on the original-resolution ODIR-5K release.
  - Keyword vocabulary (`results/audit/vocabulary.csv`) and the reviewed per-eye keyword mapping (`prereg/keyword_mapping.csv`) are committed.
  - Image sweep with two-stage duplicate detection done; duplicate threshold set in `configs/audit.yaml`.
- **Next:** the full audit run, then the Stage 1 pre-registration (registered on OSF before any model predicts a diseased eye).
- No model results yet.

## Setup

Python 3.12, managed with [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run ruff check
uv run ruff format --check
uv run pytest
```

## Data

ODIR-5K is not redistributed here. Place the original-resolution release (not a resized mirror) at:

```
data/raw/odir/annotations.xlsx
data/raw/odir/images/        # {id}_left.jpg, {id}_right.jpg
```

`data/` is gitignored. The repository commits only aggregate tables and ID-only files (splits, exclusions), never images, the annotation sheet or per-patient fields.

## Audit

```
uv run python -m rra.audit vocabulary   # keyword vocabulary from the sheet
uv run python -m rra.audit sweep        # one pass over all images: FOV, quality, duplicates (label-free)
uv run python -m rra.audit run          # exclusions, cohorts, laterality, demographics, camera, quality, power, split feasibility
```

Settings live in `configs/audit.yaml`; override any of them with `key=value` arguments. Per-patient outputs go to `data/derived/`; aggregate tables and a run record (`run_<stage>.json`: config, git SHA, package versions) go to `results/audit/`.

## Layout

```
src/rra/data/       ODIR reader, per-eye manifest, FOV geometry
src/rra/audit/      Phase 1 audit modules and CLI
src/rra/analysis/   power / MDE, bootstrap, AUC
configs/ prereg/ results/ splits/ figures/ kaggle/ paper/ tests/
```
