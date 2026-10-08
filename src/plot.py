"""Plot ABX accuracy per layer, one panel per model, as in internal_tools/tutorials/2_activation_analyses.ipynb.

Reads <results dir>/<model>/<item>/abx_*.csv written by run_abx.py (one row per layer and contrast):
the line is the mean accuracy over contrasts (= the overall ABX accuracy), the band its 95% CI.
By default draws one figure per evaluation set (item) with every model that has results for it,
to <results dir>/figures/abx_<item>.png.

Usage:
    python -m src.plot
    python -m src.plot --results_dir results/fast
    python -m src.plot --models wav2vec2-base hubert-base-ls960 --items triphone-dev-clean
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import MODELS, RESULTDIR

CONDITION_STYLES = {
    "within-speaker": dict(color=sns.color_palette("Paired")[3]),
    "across-speaker": dict(color=sns.color_palette("Paired")[2]),
    "random": dict(color="grey", dashes=(1, 2)),
}


def load_results(model_names, item="triphone-dev-clean", results_root=RESULTDIR):
    """Return {model name: DataFrame of all its abx_*.csv results for the item file}."""
    results = {}
    for model_name in model_names:
        results_dir = Path(results_root) / model_name / item
        files = sorted(results_dir.glob("abx_*.csv"))
        if not files:
            raise FileNotFoundError(f"No abx_*.csv results in {results_dir}")
        results[model_name] = pd.concat([pd.read_csv(f) for f in files])
    return results


def plot_ABX_results(ABX_results: dict, title="", filepath=None, show=True):
    """One panel per model, one line per condition present in that model's results."""
    sns.set_style("whitegrid")
    mpl.rcParams["axes.linewidth"] = 2
    mpl.rcParams["axes.edgecolor"] = "0.9"

    model_names = list(ABX_results)
    # panel width proportional to the number of layers, so e.g. CPC (CNN, LSTM) gets a narrow panel
    n_layers = [ABX_results[m]["layer"].nunique() for m in model_names]
    width_ratios = [max(n, 2) / max(n_layers) for n in n_layers]

    fig, axs = plt.subplots(
        1,
        len(model_names),
        figsize=(2.25 * sum(width_ratios), 3),
        width_ratios=width_ratios,
        sharey=True,
        squeeze=False,
    )
    for ax, model_name in zip(axs[0], model_names):
        results = ABX_results[model_name]
        for condition, style in CONDITION_STYLES.items():
            condition_results = results[results["condition"] == condition]
            if not condition_results.empty:
                sns.lineplot(
                    condition_results,
                    x="depth",
                    y="accuracy",
                    ax=ax,
                    label=condition,
                    **style,
                )
        layer_depths = dict(zip(results["layer"], results["depth"]))
        ax.set_xticks(
            list(layer_depths.values()), list(layer_depths.keys()), rotation=90
        )
        ax.set_xlim(min(layer_depths.values()), max(layer_depths.values()))
        ax.set_xlabel("")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0.45, 1)
        ax.legend().remove()
        ax.set_title(model_name, ha="left", x=0)

    # one legend for the whole figure, with every condition that appears in any panel
    handles = {}
    for ax in axs[0]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    # legend just above the panels, title above the legend (so they never overlap on narrow figures)
    fig.legend(
        handles.values(),
        handles.keys(),
        frameon=False,
        ncol=len(handles),
        loc="lower right",
        bbox_to_anchor=(1, 0.98),
    )
    if title:
        fig.suptitle(title.upper(), ha="left", x=0, y=1.12, fontweight="bold")
    if filepath:
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(filepath, dpi=300, facecolor="white", bbox_inches="tight")
    if show:
        plt.show()


def find_results(results_root=RESULTDIR):
    """Return {item: [model names with results for it]}, models in MODELS order, then the rest."""
    order = [Path(m).name for m in MODELS]
    found = {}
    for csv in sorted(Path(results_root).glob("*/*/abx_*.csv")):
        model_name, item = csv.parent.parent.name, csv.parent.name
        found.setdefault(item, set()).add(model_name)
    rank = lambda m: (order.index(m) if m in order else len(order), m)
    return {item: sorted(models, key=rank) for item, models in sorted(found.items())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=RESULTDIR,
        help="Where run_abx.py wrote the results",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model folders to plot; all with results by default",
    )
    parser.add_argument(
        "--items",
        nargs="+",
        default=None,
        help="Evaluation sets (item names) to plot; all by default",
    )
    parser.add_argument(
        "--title",
        default="ABX tests, trained models",
        help="Figure title, followed by the item name",
    )
    args = parser.parse_args()

    for item, model_names in find_results(args.results_dir).items():
        if args.items and item not in args.items:
            continue
        if args.models:
            model_names = [m for m in model_names if m in args.models]
        if not model_names:
            continue
        filepath = args.results_dir / "figures" / f"abx_{item}.png"
        plot_ABX_results(
            load_results(model_names, item, args.results_dir),
            title=f"{args.title}, {item}",
            filepath=filepath,
            show=False,
        )
        plt.close("all")
        print(f"Saved {filepath}")
