#!/bin/bash

# Example launcher for the prompt-domain analysis experiment.
# Adjust the env activation and CLI options to match your workstation.

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch-nightly

PYTHON_BIN=${PYTHON_BIN:-python}
EXPERIMENT=${EXPERIMENT:-/mnt/local-data/workspace/codes/fine-tune-clip/exps/prompt_domain_experiment.py}

# Override Hydra cfg entries from the shell to sweep domains/prompts, etc.
# Example toggles below assume the default config shipped with the repo.

${PYTHON_BIN} ${EXPERIMENT} \
  domains[0].domain_name=clipart \
  domains[1].domain_name=real \
  prompts.templates='["a photo of a {}","a sketch of a {}","an artistic drawing of a {}","a minimal icon of a {}","a realistic studio shot of a {}","a {}, high-resolution, detailed"]' \
  evaluation.max_samples_per_domain=2048 \
  evaluation.compute_topk='[1,5]' \
  output.save_json=true \
  output.json_path=outputs/prompt_vs_vision/latest.json
