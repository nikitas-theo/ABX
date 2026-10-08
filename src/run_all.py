"""Run the whole pipeline for every model: extract activations, run the ABX tasks, delete the activations.

For each model and evaluation set (see EVALS), one at a time (so at most one model's activations for
one set are on disk, ~44 GB for LibriSpeech dev-clean at float32, a few MB for the prosodic sets):
    1. extract.py  -> activations/<model>/<layer>/<file id>.pt
    2. run_abx.py  -> results/<model>/<item file name>/abx_<task>.csv, for every task of the set
    3. delete activations/<model>/
Then plot with plot.py (one figure per evaluation set, with all models).

Prepare the audio and item files first with src/prepare_tasks.py.
A model/set whose results already exist is skipped, so the script can be re-run after a failure.

With --fast, a quick dry run to check that everything works (e.g. on CPU): every evaluation set is
cut down to a few files (items/fast/), only each model's first and last layer are scored, and results
go to results/fast/.

Usage:
    python -m src.run_all
    python -m src.run_all --models facebook/wav2vec2-base cpc --evals triphone-dev-clean stress
    python -m src.run_all --fast
"""

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
# evaluation set: audio, its file format, the item file, and the ABX tasks to run on it
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
FAST_RESULTDIR = RESULTDIR / "fast"


def read_items(path_items):
    """Read an item file (.item: space separated, .csv: comma separated) with values kept as written."""
    separator = " " if Path(path_items).suffix == ".item" else ","
    return pl.read_csv(path_items, separator=separator, infer_schema=False), separator


def make_fast_items(path_items):
    """Write a small subset of an item file to items/fast/, for a quick dry run of the pipeline."""
    items, separator = read_items(path_items)
    if "phone_sequence" in items.columns:
        # prosodic: 2 words with every speaker (A, B and X say the same word, X by another speaker)
        words = sorted(items["phone_sequence"].unique())[:2]
        items = items.filter(pl.col("phone_sequence").is_in(words))
    else:
        # zero_shot: 3 files from each of 2 speakers, enough for within- and across-speaker cells
        speakers = sorted(items["speaker"].unique())[:2]
        files = [
            f
            for s in speakers
            for f in sorted(items.filter(pl.col("speaker") == s)["#file"].unique())[:3]
        ]
        items = items.filter(pl.col("#file").is_in(files))
    fast_path = ITEMDIR / "fast" / Path(path_items).name
    fast_path.parent.mkdir(parents=True, exist_ok=True)
    items.write_csv(fast_path, separator=separator)
    return fast_path


def run_model_on_eval(
    model_name_or_path, eval_name, batch_size=16, keep_activations=False, fast=False
):
    model_name = Path(
        model_name_or_path
    ).name  # folder name used by extract.py and run_abx.py
    spec = EVALS[eval_name]
    path_items = make_fast_items(spec["path_items"]) if fast else spec["path_items"]
    results_root = FAST_RESULTDIR if fast else RESULTDIR
    results_dir = results_root / model_name / path_items.stem
    if all((results_dir / f"abx_{task}.csv").exists() for task in spec["tasks"]):
        print(f"Skipping {model_name} on {eval_name}: results already in {results_dir}")
        return

    try:
        extract(
            model_name_or_path,
            spec["audio_dir"],
            file_format=spec["file_format"],
            out_dir=ACTIVATIONDIR,
            batch_size=batch_size,
            # only the audio files the item file uses
            file_ids=set(read_items(path_items)[0]["#file"]),
        )
        gc.collect()
        torch.cuda.empty_cache()
        layers = None  # all
        if fast:
            info = json.loads((ACTIVATIONDIR / model_name / "info.json").read_text())
            layers = [info["layers"][0], info["layers"][-1]]
        evaluate_abx(
            model_name,
            path_items,
            spec["tasks"],
            activations_dir=ACTIVATIONDIR,
            layers=layers,
            results_dir=results_root,
        )
    finally:
        # delete the activations even if a step failed, so a failure never leaves ~44 GB behind
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
        default=list(EVALS),
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
        "--fast",
        action="store_true",
        help="Quick dry run: a few files per evaluation set, first and last layer, results in results/fast/",
    )
    args = parser.parse_args()

    # keep going if one model fails (e.g. a missing checkpoint), and report all failures at the end
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
