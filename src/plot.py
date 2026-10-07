import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import RESULTDIR

CONDITION_STYLES = {
    "within-speaker": dict(color=sns.color_palette("Paired")[3]),
    "across-speaker": dict(color=sns.color_palette("Paired")[2]),
    "random": dict(color="grey", dashes=(1, 2)),
}


def load_results(model_names, item="triphone-dev-clean"):
    """Return {model name: DataFrame of all its abx_*.csv results for the item file}."""
    results = {}
    for model_name in model_names:
        results_dir = RESULTDIR / model_name / item
        files = sorted(results_dir.glob("abx_*.csv"))
        if not files:
            raise FileNotFoundError(f"No abx_*.csv results in {results_dir}")
        results[model_name] = pd.concat([pd.read_csv(f) for f in files])
    return results


def plot_ABX_results(ABX_results: dict, title="", filepath=None):
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
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_names", nargs="+", help="Model folders in results/, e.g. wav2vec2-base"
    )
    parser.add_argument(
        "--item",
        default="triphone-dev-clean",
        help="Item file name (results subfolder)",
    )
    parser.add_argument("--title", default="", help="Figure title")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Defaults to results/figures/abx_<item>.png",
    )
    args = parser.parse_args()

    out = args.out or RESULTDIR / "figures" / f"abx_{args.item}.png"
    plot_ABX_results(
        load_results(args.model_names, args.item), title=args.title, filepath=out
    )
