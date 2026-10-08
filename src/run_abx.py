import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch

from src.config import ACTIVATIONDIR, RESULTDIR
from src.tasks import get_task, TASK_NAMES


def evaluate_abx(
    model_name: str,
    path_items: str | Path,
    tasks: list[str],
    activations_dir: str | Path = ACTIVATIONDIR,
    layers: list[str] | None = None,
    results_dir: str | Path = RESULTDIR,
):
    """
    Evaluate saved activations on ABX tasks, loading each layer's features once for all tasks.

    Args:
        model_name (str): The model's folder in activations_dir (e.g. "wav2vec2-base").
        path_items (str | Path): The path to the items file for the ABX tasks.
        tasks (list[str]): Tasks from TASK_NAMES (see src/tasks.py) to run on this item file.
        activations_dir (str | Path): Where extract.py saved the activations.
        layers (list[str] | None): Only score these layers; all extracted layers if None.
        results_dir (str | Path): Results go to <results_dir>/<model>/<item file name>/abx_<task>.csv.
    """
    model_dir = Path(activations_dir) / model_name
    info = json.loads((model_dir / "info.json").read_text())
    results = {task: [] for task in tasks}

    for depth, layer in enumerate(info["layers"]):
        if layers is not None and layer not in layers:
            continue
        datasets = {}  # one load per loader: tasks on the same item file share the features
        for task in tasks:
            condition, loader, task_fn = get_task(task)
            if loader not in datasets:
                datasets[loader] = loader(path_items, model_dir / layer, frequency=info["frequency"])
            pair_scores = task_fn(datasets[loader])
            # one row per contrast (e.g. phone pair A, B) with its labels as returned by the task;
            # the mean accuracy over rows is the overall ABX accuracy
            for pair in pair_scores:
                labels = {k: v for k, v in pair.items() if k != "score"}
                results[task].append(
                    {
                        "model": model_name,
                        "condition": condition,
                        "layer": layer,
                        "depth": depth,
                        **labels,
                        "error_rate": pair["score"],
                        "accuracy": 1 - pair["score"],
                        "item": str(path_items),
                        "task": task,
                    }
                )
            error_rate = sum(pair["score"] for pair in pair_scores) / len(pair_scores)
            print(f"{layer} {task}: accuracy {1 - error_rate:.4f}")
        # clean: free this layer's features on the GPU before the next layer is loaded
        del datasets
        gc.collect()
        torch.cuda.empty_cache()

    # save results, one file per task
    for task, rows in results.items():
        results_path = Path(results_dir) / model_name / Path(path_items).stem / f"abx_{task}.csv"
        results_path.parent.mkdir(exist_ok=True, parents=True)
        pd.DataFrame(rows).to_csv(results_path, index=False)
        print(f"Saved results to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        type=str,
        help="Model folder in the activations directory, e.g. wav2vec2-base",
    )
    parser.add_argument("path_items", type=str, help="Path to the items file")
    parser.add_argument("tasks", type=str, nargs="+", choices=TASK_NAMES, help="ABX tasks to run")
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
        tasks=args.tasks,
        activations_dir=args.activations_dir,
    )
