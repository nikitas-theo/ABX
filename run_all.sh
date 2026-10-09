#!/usr/bin/env bash
# The whole pipeline: setup, data, ABX for every model, plots (results/, results/figures/).
set -euo pipefail
cd "$(dirname "$0")"

# 1. once: environment, internal_tools, and all model weights
python3 setup_project.py --models facebook/wav2vec2-base microsoft/wavlm-base \
    facebook/hubert-base-ls960 MarvinLvn/BabyHuBERT cpc melhubert

# 2. once: audio + item files for both benchmarks
uv run python -m src.prepare_tasks zero_shot prosodic

# 3. every model on the ZeroSpeech samples and the stress sets
#    (the full splits: --evals triphone-dev-clean triphone-dev-other);
#    a failing model doesn't stop the others, failures are listed and make this exit 1
status=0
uv run python -m src.run_all || status=$?

# 4. one figure per evaluation set, plus all of them stacked in abx_all.png
uv run python -m src.plot
exit $status
