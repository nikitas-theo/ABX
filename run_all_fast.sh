#!/usr/bin/env bash
# Quick dry run of the whole pipeline, e.g. on CPU: a few files per evaluation set (items/fast/),
# each model's first and last layer only, results in results/fast/.
set -euo pipefail
cd "$(dirname "$0")"

# 1. once: environment, internal_tools, and all model weights
python3 setup_project.py --models facebook/wav2vec2-base microsoft/wavlm-base \
    facebook/hubert-base-ls960 MarvinLvn/BabyHuBERT cpc melhubert

# 2. once: audio + item files for both benchmarks
uv run python -m src.prepare_tasks zero_shot prosodic

# 3. every model on a few files of each evaluation set
#    (a failing model doesn't stop the others; failures are listed and make this exit 1)
status=0
uv run python -m src.run_all --fast || status=$?

# 4. plots of the models that finished
uv run python -m src.plot --results_dir results/fast
exit $status
