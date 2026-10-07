# triphone within- and across-speaker on clean and other

from fastabx import Dataset, Task, Score, Subsampler

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
        case _:
            raise ValueError(f"Unknown task: {task}")


TASK_NAMES = ["zero_shot_triphone_within", "zero_shot_triphone_across"]


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
