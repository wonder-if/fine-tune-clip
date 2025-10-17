# Experiment Layout Guide

This repo separates reproducible experiments, hydra configuration, and shell entry points so you can iterate on research ideas without mixing them with training code or evaluation utilities.

## Key directories

- `clip_tuner/configs/`: Hydra config tree. Each experiment picks a root config (for zero-shot baselines, prompt-domain study, etc.) and you override specific fields from the CLI or scripts.
- `exps/`: Python entry points that define the actual experiment logic. Each file should be a small Hydra program that pulls a model, builds datasets, runs evaluation/training, and prints structured results.
- `scripts/`: Thin bash wrappers for launching experiments, handling environment activation, sweeps, and result aggregation. These should only shell out to the `exps/` modules.
- `outputs/`: Collected metrics, JSON dumps, CSV summaries, and other artifacts produced by the scripts.
- `docs/`: Lightweight notes like this guide so you remember how everything fits together.

## Adding a new experiment

1. Create or reuse a Hydra config under `clip_tuner/configs/`. Favor config groups (e.g. `data/`, `prompts/`) so components are easy to swap.
2. Drop a Python driver in `exps/` that imports `clip_tuner` utilities, reads the config, and prints machine-readable results (JSON is easiest for downstream scripts).
3. Add a launcher under `scripts/` if the experiment needs shell automation (domain sweeps, repeated runs, etc.).
4. Keep dataset/model assumptions in the config—avoid hardcoding paths in the Python entry point.
5. Document any special instructions here or in `README.md` so future you knows how to relaunch the study.

This structure keeps experiments reproducible, makes it obvious where to make changes, and leaves `tests/` free for automated checks.
