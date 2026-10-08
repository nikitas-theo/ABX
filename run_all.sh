# 1. once: environment, internal_tools, and all model weights
python3 setup_project.py --models facebook/wav2vec2-base microsoft/wavlm-base \
    facebook/hubert-base-ls960 MarvinLvn/BabyHuBERT cpc melhubert

# 2. once: audio + item files for both benchmarks
uv run python -m src.prepare_tasks zero_shot prosodic

# 3. extract and score every model on every evaluation set
uv run python -m src.run_all

# 4. one figure per evaluation set, with all models -> results/figures/
uv run python -m src.plot
