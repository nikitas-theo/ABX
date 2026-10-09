"""For each model and evaluation set (see EVALS): extract the activations (extract.py), run the
set's ABX tasks (run_abx.py), then delete the activations, so only one model/set is on disk at a time.

Tasks whose results already exist are skipped, so a failed run can simply be restarted.
--fast is a quick dry run: a few files per set, first and last layer, results in results/fast/.
Prepare the data first with src/prepare_tasks.py; plot with src/plot.py.

Usage:
    python -m src.run_all
    python -m src.run_all --models facebook/wav2vec2-base cpc --evals triphone-dev-clean stress
    python -m src.run_all --fast
"""

import os

# less GPU memory fragmentation when each layer's features are loaded (set before importing torch)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import json
import shutil
import traceback
from pathlib import Path

import polars as pl
import torch

from src.config import ACTIVATIONDIR, DATADIR, ITEMDIR, MODELS, RESULTDIR
from src.extract import extract
from src.run_abx import evaluate_abx

ZEROSPEECH_ITEMS = ITEMDIR / "zerospeech2021-triphone" / "item"
ZERO_SHOT_TASKS = [
    "zero_shot_triphone_within",
    "zero_shot_triphone_across",
    "zero_shot_triphone_random",
]
EVALS = {
    **{
        f"triphone-{split}": dict(
            audio_dir=DATADIR / "LibriSpeech" / split,
            file_format="flac",
            path_items=ZEROSPEECH_ITEMS / f"triphone-{split}.item",
            tasks=ZERO_SHOT_TASKS,
        )
        for split in ["dev-clean", "dev-other"]
    },
    # ~10% samples of the splits, see prepare_tasks.py
    **{
        f"triphone-{split}-sample": dict(
            audio_dir=DATADIR / "LibriSpeech" / split,
            file_format="flac",
            path_items=ZEROSPEECH_ITEMS.parent
            / "sample"
            / f"triphone-{split}-sample.item",
            tasks=ZERO_SHOT_TASKS,
        )
        for split in ["dev-clean", "dev-other"]
    },
    **{
        name: dict(
            audio_dir=DATADIR / "prosodic" / name,
            file_format="wav",
            path_items=ITEMDIR / "prosodic" / f"{name}.csv",
            tasks=["prosodic_across", "prosodic_random"],
        )
        for name in ["stress", "stress_syn", "stress_kokoro"]
    },
}
DEFAULT_EVALS = [
    "triphone-dev-clean-sample",
    "triphone-dev-other-sample",
    "stress",
    "stress_syn",
    "stress_kokoro",
]
FAST_RESULTDIR = RESULTDIR / "fast"


def separator(path_items):
    return " " if Path(path_items).suffix == ".item" else ","


def read_items(path_items):
    """An item file with its values kept as written."""
    return pl.read_csv(path_items, separator=separator(path_items), infer_schema=False)


def make_fast_items(path_items):
    """Write a small subset of an item file to items/fast/, for a dry run."""
    items = read_items(path_items)
    if "phone_sequence" in items.columns:
        # prosodic: 2 words with every speaker
        words = sorted(items["phone_sequence"].unique())[:2]
        items = items.filter(pl.col("phone_sequence").is_in(words))
    else:
        # zero_shot: 3 files from each of 2 speakers
        speakers = sorted(items["speaker"].unique())[:2]
        files = [
            f
            for s in speakers
            for f in sorted(items.filter(pl.col("speaker") == s)["#file"].unique())[:3]
        ]
        items = items.filter(pl.col("#file").is_in(files))
    fast_path = ITEMDIR / "fast" / Path(path_items).name
    fast_path.parent.mkdir(parents=True, exist_ok=True)
    items.write_csv(fast_path, separator=separator(path_items))
    return fast_path


def select_layers(layers, every_n_layers=1):
    """Every n-th layer, always including the first and the last."""
    selected = layers[::every_n_layers]
    if layers[-1] not in selected:
        selected.append(layers[-1])
    return selected


def run_model_on_eval(
    model_name_or_path,
    eval_name,
    batch_size=16,
    keep_activations=False,
    fast=False,
    every_n_layers=1,
):
    model_name = Path(model_name_or_path).name
    spec = EVALS[eval_name]
    path_items = make_fast_items(spec["path_items"]) if fast else spec["path_items"]
    results_root = FAST_RESULTDIR if fast else RESULTDIR
    results_dir = results_root / model_name / path_items.stem
    tasks = [t for t in spec["tasks"] if not (results_dir / f"abx_{t}.csv").exists()]
    if not tasks:
        print(f"Skipping {model_name} on {eval_name}: results already in {results_dir}")
        return

    try:
        extract(
            model_name_or_path,
            spec["audio_dir"],
            file_format=spec["file_format"],
            out_dir=ACTIVATIONDIR,
            batch_size=batch_size,
            file_ids=set(read_items(path_items)["#file"]),
        )
        gc.collect()
        torch.cuda.empty_cache()
        info = json.loads((ACTIVATIONDIR / model_name / "info.json").read_text())
        if fast:
            layers = [info["layers"][0], info["layers"][-1]]
        else:
            layers = select_layers(info["layers"], every_n_layers)
        evaluate_abx(
            model_name,
            path_items,
            tasks,
            activations_dir=ACTIVATIONDIR,
            layers=layers,
            results_dir=results_root,
        )
    finally:
        # also on failure, so no activations are left behind
        if not keep_activations:
            shutil.rmtree(ACTIVATIONDIR / model_name, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="Models as accepted by src/models.py",
    )
    parser.add_argument(
        "--evals",
        nargs="+",
        default=DEFAULT_EVALS,
        choices=list(EVALS),
        help="Evaluation sets to run",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Files per forward pass in extraction",
    )
    parser.add_argument(
        "--keep_activations",
        action="store_true",
        help="Do not delete activations afterwards",
    )
    parser.add_argument(
        "--every_n_layers",
        type=int,
        default=1,
        help="Score every n-th layer (always including the first and last), e.g. 2 for about half",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick dry run: a few files per evaluation set, first and last layer, results in results/fast/",
    )
    args = parser.parse_args()

    # keep going if one model fails, and report the failures at the end
    failed = []
    for model_name_or_path in args.models:
        for eval_name in args.evals:
            print(f"\n=== {model_name_or_path} on {eval_name} ===")
            try:
                run_model_on_eval(
                    model_name_or_path,
                    eval_name,
                    args.batch_size,
                    args.keep_activations,
                    args.fast,
                    args.every_n_layers,
                )
            except Exception:
                traceback.print_exc()
                failed.append(f"{model_name_or_path} on {eval_name}")

    if failed:
        print("\nFailed:\n  " + "\n  ".join(failed))
        raise SystemExit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()
