import argparse
import gc
import json
from pathlib import Path

import pandas as pd

from src.config import ACTIVATIONDIR, RESULTDIR
from src.tasks import get_task, TASK_NAMES


def evaluate_abx(
    model_name: str,
    path_items: str | Path,
    task: str,
    activations_dir: str | Path = ACTIVATIONDIR,
):
    """
    Evaluate saved activations on the ABX task.

    Args:
        model_name (str): The model's folder in activations_dir (e.g. "wav2vec2-base").
        path_items (str | Path): The path to the items file for the ABX task.
        task (str): One of TASK_NAMES (see src/tasks.py).
        activations_dir (str | Path): Where extract.py saved the activations.
    """
    model_dir = Path(activations_dir) / model_name
    info = json.loads((model_dir / "info.json").read_text())
    condition, task_fn = get_task(task)

    results = []
    results_path = RESULTDIR / model_name / Path(path_items).stem / f"abx_{task}.csv"
    for depth, layer in enumerate(info["layers"]):
        pair_scores = task_fn(
            path_items, model_dir / layer, frequency=info["frequency"]
        )
        # one row per phone pair (A, B); the mean accuracy over pairs is the overall ABX accuracy
        for pair in pair_scores:
            results.append(
                {
                    "model": model_name,
                    "condition": condition,
                    "layer": layer,
                    "depth": depth,
                    "phone_a": pair["#phone"],
                    "phone_b": pair["#phone_b"],
                    "size": pair["size"],
                    "error_rate": pair["score"],
                    "accuracy": 1 - pair["score"],
                    "item": str(path_items),
                    "task": task,
                }
            )
        error_rate = sum(pair["score"] for pair in pair_scores) / len(pair_scores)
        print(f"{layer}: accuracy {1 - error_rate:.4f}")
        # clean
        gc.collect()

    # save results
    results_path.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        type=str,
        help="Model folder in the activations directory, e.g. wav2vec2-base",
    )
    parser.add_argument("path_items", type=str, help="Path to the items file")
    parser.add_argument("task", type=str, choices=TASK_NAMES, help="ABX task to run")
    parser.add_argument(
        "--activations_dir",
        type=str,
        default=ACTIVATIONDIR,
        help="Where extract.py saved the activations",
    )
    args = parser.parse_args()
    evaluate_abx(
        args.model_name,
        path_items=args.path_items,
        task=args.task,
        activations_dir=args.activations_dir,
    )
