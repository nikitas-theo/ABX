# triphone within- and across-speaker on clean and other

import math
from decimal import Decimal
from pathlib import Path

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
    """Return (condition, task function) for a task name.

    Task functions return one row per phone pair (A, B): {"#phone", "#phone_b", "score", "size"},
    where score is that pair's ABX error rate. Their mean is the overall ZeroSpeech error rate.
    """
    match task:
        case "zero_shot_triphone_across":
            return "across-speaker", zero_shot_triphone_across
        case "zero_shot_triphone_within":
            return "within-speaker", zero_shot_triphone_within
        case "prosodic_across":
            return "across-speaker", prosodic_across
        case "zero_shot_triphone_random":
            return "random", zero_shot_triphone_random
        case "prosodic_random":
            return "random", prosodic_random
        case _:
            raise ValueError(f"Unknown task: {task}")


TASK_NAMES = [
    "zero_shot_triphone_within",
    "zero_shot_triphone_across",
    "prosodic_across",
    "zero_shot_triphone_random",
    "prosodic_random",
]


def zero_shot_triphone_within(path_items, features_dir, frequency=50):
    # features_dir holds one <file id>.pt of shape (n_frames, dim) per audio file
    dataset = Dataset.from_item(path_items, features_dir, frequency=frequency)
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


def zero_shot_triphone_across(path_items, features_dir, frequency=50):
    # features_dir holds one <file id>.pt of shape (n_frames, dim) per audio file
    dataset = Dataset.from_item(path_items, features_dir, frequency=frequency)
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


def prosodic_across(path_items, features_dir, frequency=50):
    """Prosodic ABX (https://arxiv.org/abs/2604.02102), as in the prosodic-abx repo's run_abx.py."""
    dataset = load_dataset_clamped(path_items, features_dir, frequency)
    # ON the stress pattern, BY the word (so only stress differs);
    # A and B from one speaker, X from a different speaker. No subsampling.
    task = Task(dataset, on="accent_pattern", by=["phone_sequence"], across=["speaker"])
    score = Score(task, "angular")
    # average over speakers, then over (word, contrast)
    return score.details(levels=["speaker"]).to_dicts()


def load_dataset_clamped(path_items, features_dir, frequency):
    """Like Dataset.from_item, but a segment that runs past the end of its features is cut at the last frame.

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


# --- random baselines, like the "random" triplets in internal_tools/tutorials/2_activation_analyses.ipynb ---


def shuffle_labels(dataset, on, by):
    """Return the dataset with the ON labels shuffled within each BY group.

    The task then has the same cells and cell sizes as the real one, but whether A and X share a
    category is random, so the expected accuracy is chance (0.5).
    """
    labels = dataset.labels.with_columns(pl.col(on).shuffle(seed=SEED).over(by))
    return Dataset(labels=labels, accessor=dataset.accessor)


def zero_shot_triphone_random(path_items, features_dir, frequency=50):
    """Control for zero_shot_triphone_within: the same task with #phone shuffled within each cell."""
    dataset = Dataset.from_item(path_items, features_dir, frequency=frequency)
    by = ["speaker", "next-phone", "prev-phone"]
    task = Task(
        shuffle_labels(dataset, "#phone", by),
        on="#phone",
        by=by,
        subsampler=Subsampler(
            max_size_group=MAX_SIZE_GROUP, max_x_across=None, seed=SEED
        ),
    )
    score = Score(task, "angular")
    return score.details(levels=LEVELS).to_dicts()


def prosodic_random(path_items, features_dir, frequency=50):
    """Control for prosodic_across: the same task with the stress labels shuffled within each word."""
    dataset = load_dataset_clamped(path_items, features_dir, frequency)
    task = Task(
        shuffle_labels(dataset, "accent_pattern", ["phone_sequence"]),
        on="accent_pattern",
        by=["phone_sequence"],
        across=["speaker"],
    )
    score = Score(task, "angular")
    return score.details(levels=["speaker"]).to_dicts()
