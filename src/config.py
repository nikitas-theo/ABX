"""Paths and the model list shared by the scripts (paths are relative to the repo root)."""

from pathlib import Path

MODELDIR = Path("models")
DATADIR = Path("data")
ACTIVATIONDIR = Path("activations")
RESULTDIR = Path("results")
ITEMDIR = Path("items")

# the models of internal_tools/tutorials/2_activation_analyses.ipynb, in its plot order
MODELS = [
    "facebook/wav2vec2-base",
    "microsoft/wavlm-base",
    "spidr",
    "facebook/hubert-base-ls960",
    "MarvinLvn/BabyHuBERT",
    "melhubert",
    "cpc",
]
