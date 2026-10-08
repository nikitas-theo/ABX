"""Run the whole pipeline for every model: extract activations, run the ABX tasks, delete the activations.

For each model and evaluation set (see EVALS), one at a time (so at most one model's activations for
one set are on disk, ~44 GB for LibriSpeech dev-clean at float32, a few MB for the prosodic sets):
    1. extract.py  -> activations/<model>/<layer>/<file id>.pt
    2. run_abx.py  -> results/<model>/<item file name>/abx_<task>.csv, for every task of the set
    3. delete activations/<model>/
Finally plot.py draws one figure per evaluation set with all models.

Prepare the audio and item files first with src/prepare_tasks.py.
A model/set whose results already exist is skipped, so the script can be re-run after a failure.

Usage:
    python -m src.run_all
    python -m src.run_all --models facebook/wav2vec2-base cpc --evals triphone-dev-clean stress
"""

import argparse
import gc
import shutil
import traceback
from pathlib import Path

import torch

from src.config import ACTIVATIONDIR, DATADIR, ITEMDIR, RESULTDIR
from src.extract import extract
from src.plot import load_results, plot_ABX_results
from src.run_abx import evaluate_abx

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

ZEROSPEECH_ITEMS = ITEMDIR / "zerospeech2021-triphone" / "item"
ZERO_SHOT_TASKS = ["zero_shot_triphone_within", "zero_shot_triphone_across"]
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
            tasks=["prosodic_across"],
        )
        for name in ["stress", "stress_syn", "stress_kokoro"]
    },
}


def run_model_on_eval(
    model_name_or_path, eval_name, batch_size=16, keep_activations=False
):
    model_name = Path(
        model_name_or_path
    ).name  # folder name used by extract.py and run_abx.py
    spec = EVALS[eval_name]
    results_dir = RESULTDIR / model_name / spec["path_items"].stem
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
        )
        gc.collect()
        torch.cuda.empty_cache()
        for task in spec["tasks"]:
            evaluate_abx(
                model_name, spec["path_items"], task, activations_dir=ACTIVATIONDIR
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
                )
            except Exception:
                traceback.print_exc()
                failed.append(f"{model_name_or_path} on {eval_name}")

    # one figure per evaluation set, with every model that has results for it
    for eval_name in args.evals:
        item = EVALS[eval_name]["path_items"].stem
        model_names = [
            Path(m).name
            for m in args.models
            if list((RESULTDIR / Path(m).name / item).glob("abx_*.csv"))
        ]
        if model_names:
            plot_ABX_results(
                load_results(model_names, item),
                title=f"ABX tests, trained models, {eval_name}",
                filepath=RESULTDIR / "figures" / f"abx_{item}.png",
            )

    if failed:
        print("\nFailed:\n  " + "\n  ".join(failed))
        raise SystemExit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()
