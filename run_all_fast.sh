#!/usr/bin/env bash
# Quick dry run of the whole pipeline (works on CPU): every evaluation set is cut down to a few
# files (items/fast/), only each model's first and last layer are scored, results go to results/fast/.
set -euo pipefail
cd "$(dirname "$0")"

# 1. once: environment, internal_tools, and all model weights
python3 setup_project.py --models facebook/wav2vec2-base microsoft/wavlm-base \
    facebook/hubert-base-ls960 MarvinLvn/BabyHuBERT cpc melhubert

# 2. once: audio + item files for both benchmarks
uv run python -m src.prepare_tasks zero_shot prosodic

# 3. extract and score every model on a few files of every evaluation set
#    (a failing model doesn't stop the others; failures are listed at the end and make this exit 1)
status=0
uv run python -m src.run_all --fast || status=$?

# 4. one figure per evaluation set, with all models that finished -> results/fast/figures/
uv run python -m src.plot --results_dir results/fast
exit $status
