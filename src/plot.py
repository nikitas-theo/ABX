from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


CONDITION_STYLES = {
    "within-speaker": dict(color=sns.color_palette("Paired")[3]),
    "across-speaker": dict(color=sns.color_palette("Paired")[2]),
    "random": dict(color="grey", dashes=(1, 2)),
}


def plot_ABX_results(results: pd.DataFrame, title="", filepath=None):
    """Plot ABX accuracy per layer for a single model, one line per condition in `results`.

    `results` is the CSV written by abx.py: columns condition, layer, depth, accuracy.
    """
    sns.set_style("whitegrid")
    mpl.rcParams["axes.linewidth"] = 2
    mpl.rcParams["axes.edgecolor"] = "0.9"

    fig, ax = plt.subplots(figsize=(4.5, 3))

    for condition, style in CONDITION_STYLES.items():
        condition_results = results[results["condition"] == condition]
        if condition_results.empty:
            continue
        sns.lineplot(
            condition_results, x="depth", y="accuracy", ax=ax, label=condition, **style
        )

    layer_depths = dict(zip(results["layer"], results["depth"]))
    ax.set_xticks(list(layer_depths.values()), list(layer_depths.keys()), rotation=90)
    ax.set_xlim(min(layer_depths.values()), max(layer_depths.values()))
    ax.set_xlabel("")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.45, 1)
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02))
    ax.set_title(results["model"].iloc[0], ha="left", x=0, y=1.12)
    if title:
        plt.suptitle(title.upper(), ha="left", x=0.095, y=1.1, fontweight="bold")
    if filepath:
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(filepath, dpi=300, facecolor="white", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    model_name = "wav2vec2-base"
    results_dir = Path("results") / model_name
    results = pd.read_csv(results_dir / "abx.csv")
    plot_ABX_results(results, filepath=results_dir / "abx.png")
