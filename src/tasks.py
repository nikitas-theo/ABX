# ABX tasks: ZeroSpeech 2021 triphone (within- and across-speaker) and Prosodic ABX (English stress),
# each with a random baseline

import math
from decimal import Decimal
from pathlib import Path

import numpy as np
import polars as pl
import torch
from fastabx import Dataset, InMemoryAccessor, Task, Score, Subsampler
from fastabx.dataset import read_labels

# ZeroSpeech 2021 subsampling: at most 10 instances of A, B or X per cell,
# and in the across-speaker case at most 5 X per (A, B)
MAX_SIZE_GROUP = 10
MAX_X_ACROSS = 5
SEED = 0
# average over contexts first, then over speakers, as in ZeroSpeech
LEVELS = [("next-phone", "prev-phone"), "speaker"]


def get_task(task: str):
    """Return (condition, dataset loader, task function) for a task name.

    The loader builds the fastabx Dataset from (path_items, features_dir, frequency); tasks with the
    same loader share one loaded dataset. Task functions take that dataset and return one row per
    contrast (e.g. phone pair A, B) with its labels, "score" (that contrast's ABX error rate) and
    "size"; the mean score over rows is the overall error rate.
    """
    match task:
        case "zero_shot_triphone_within":
            return "within-speaker", load_dataset, zero_shot_triphone_within
        case "zero_shot_triphone_across":
            return "across-speaker", load_dataset, zero_shot_triphone_across
        case "zero_shot_triphone_random":
            return "random", load_dataset, zero_shot_triphone_random
        case "prosodic_across":
            return "across-speaker", load_dataset_clamped, prosodic_across
        case "prosodic_random":
            return "random", load_dataset_clamped, prosodic_random
        case _:
            raise ValueError(f"Unknown task: {task}")


TASK_NAMES = [
    "zero_shot_triphone_within",
    "zero_shot_triphone_across",
    "zero_shot_triphone_random",
    "prosodic_across",
    "prosodic_random",
]


# --- loading features ---


def load_dataset(path_items, features_dir, frequency=50):
    """fastabx Dataset from an item file and features_dir/<file id>.pt of shape (n_frames, dim)."""
    return Dataset.from_item(path_items, features_dir, frequency=frequency)


def load_dataset_clamped(path_items, features_dir, frequency=50):
    """Like load_dataset, but a segment that runs past the end of its features is cut at the last frame.

    Whole-word items (onset 0, offset = clip duration) often end one frame past what the model's
    convolutions output, which Dataset.from_item rejects; the prosodic-abx repo clamps them like this.
    """
    labels = read_labels(path_items, "#file", "onset", "offset")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, segments, indices, pos = {}, [], {}, 0
    for i, row in enumerate(labels.iter_rows(named=True)):
        file_id = row["#file"]
        if file_id not in features:
            features[file_id] = torch.load(Path(features_dir) / f"{file_id}.pt")
        # same frontiers as fastabx's item_frontiers (in Decimal, as onset/offset are read),
        # with the end clamped to the features
        freq, half = Decimal(str(frequency)), Decimal("0.5")
        start = math.ceil(row["onset"] * freq - half)
        end = min(
            math.floor(row["offset"] * freq - half) + 1, features[file_id].shape[0]
        )
        segments.append(features[file_id][start:end])
        indices[i] = (pos, pos + end - start)
        pos += end - start
    return Dataset(
        labels=labels, accessor=InMemoryAccessor(indices, torch.cat(segments), device)
    )


# --- ZeroSpeech 2021 triphone ABX ---


def zero_shot_triphone_within(dataset):
    # A, B and X all come from the same speaker
    task = Task(
        dataset,
        on="#phone",
        by=["speaker", "next-phone", "prev-phone"],
        subsampler=Subsampler(
            max_size_group=MAX_SIZE_GROUP, max_x_across=None, seed=SEED
        ),
    )
    score = Score(task, "angular")
    return score.details(levels=LEVELS).to_dicts()


def zero_shot_triphone_across(dataset):
    # A and B from one speaker, X from a different speaker.
    task = Task(
        dataset,
        on="#phone",
        by=["next-phone", "prev-phone"],
        across=["speaker"],
        subsampler=Subsampler(
            max_size_group=MAX_SIZE_GROUP, max_x_across=MAX_X_ACROSS, seed=SEED
        ),
    )
    score = Score(task, "angular")
    return score.details(levels=LEVELS).to_dicts()


# --- Prosodic ABX (https://arxiv.org/abs/2604.02102) ---


def prosodic_across(dataset):
    """As in the prosodic-abx repo's run_abx.py."""
    # ON the stress pattern, BY the word (so only stress differs);
    # A and B from one speaker, X from a different speaker. No subsampling.
    task = Task(dataset, on="accent_pattern", by=["phone_sequence"], across=["speaker"])
    score = Score(task, "angular")
    # average over speakers, then over (word, contrast)
    return score.details(levels=["speaker"]).to_dicts()


# --- random baselines, like the "random" triplets in internal_tools/tutorials/2_activation_analyses.ipynb ---


def shuffle_labels(dataset, on, by):
    """Return the dataset with the ON labels shuffled within each BY group.

    The task then has the same cells and cell sizes as the real one, but whether A and X share a
    category is random, so the expected accuracy is chance (0.5).
    """
    # each group needs its own permutation: pl.col(on).shuffle(seed).over(by) would apply the same
    # one to every group of the same size, i.e. a fixed relabeling that biases the baseline.
    # Instead, sort the labels within each group by an independent random key per row.
    key = np.random.default_rng(SEED).random(len(dataset.labels))
    labels = (
        dataset.labels.with_columns(_shuffle_key=pl.Series(key))
        .with_columns(pl.col(on).sort_by("_shuffle_key").over(by))
        .drop("_shuffle_key")
    )
    return Dataset(labels=labels, accessor=dataset.accessor)


def zero_shot_triphone_random(dataset):
    """Control for zero_shot_triphone_within: the same task with #phone shuffled within each cell."""
    return zero_shot_triphone_within(
        shuffle_labels(dataset, "#phone", ["speaker", "next-phone", "prev-phone"])
    )


def prosodic_random(dataset):
    """Control for prosodic_across: the same task with the stress labels shuffled within each word."""
    return prosodic_across(shuffle_labels(dataset, "accent_pattern", ["phone_sequence"]))
