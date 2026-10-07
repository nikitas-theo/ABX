import tqdm
from transformers import AutoModel
from pathlib import Path
import soundfile as sf
import pandas as pd
import torch
from torch.utils.data import DataLoader
from internal_tools.dataset_utils import AnnotatedAudioDataset, aadl_collate_fn
from internal_tools.preprocessors import AudioPreprocessor
from internal_tools.extractors import AudioModelExtractor

from dotenv import load_dotenv
import gc

from fastabx import Dataset, Task, Score
from src.config import MODELDIR

import argparse

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"


# make dummy annotations file
def make_annotations(audio_dir: str | Path, file_format: str = "flac") -> pd.DataFrame:
    audio_dir = Path(audio_dir)
    rows = []
    for f in sorted(audio_dir.glob(f"**/*.{file_format}")):
        rows.append(
            {
                "file_id": str(f.relative_to(audio_dir).with_suffix("")),
                "start_time": 0.0,
                "end_time": sf.info(f).duration,
            }
        )
    return pd.DataFrame(rows)


def get_model(model_name_or_path: str):
    match model_name_or_path:
        case "facebook/wav2vec2-base":
            model = AutoModel.from_pretrained(model_name_or_path, cache_dir=MODELDIR)
            preprocessor = AudioPreprocessor.for_hf_model(model_name_or_path)
        case _:
            raise ValueError(f"Unsupported model: {model_name_or_path}")
    return model, preprocessor


def evaluate_abx(
    model_name_or_path: str,
    audio_dir: str | Path,
    file_format: str,
    path_items: str | Path,
    path_audio_flat: str | Path,
):
    """
    Evaluate the model on the ABX task.

    Args:
        model_name_or_path (str): The name or path of the model.
        audio_dir (str | Path): The directory containing the audio files.
        file_format (str): The format of the audio files (e.g., "flac").
        path_items (str | Path): The path to the items file for the ABX task.
        path_audio_flat (str | Path): The path to the flattened audio directory.
    """
    audio_dir = Path(audio_dir)
    path = Path(path_audio_flat)

    model, preprocessor = get_model(model_name_or_path)
    annotations = make_annotations(audio_dir, file_format=file_format)
    dataset = AnnotatedAudioDataset(annotations, audio_dir, file_format=file_format)
    batch_size = 8
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=aadl_collate_fn)
    extr = AudioModelExtractor(model)
    activations = {}

    for batch_data in tqdm.tqdm(
        dl, desc=f"Extracting activations for {len(dl)} batches", total=len(dl)
    ):
        inputs = preprocessor(
            batch_data["audio_signal"],
            sampling_rate=batch_data["audio_sampling_rate"],
            padding=True,
            return_tensors="pt",
            padding_side="right",
        ).to(device)

        outputs = model(inputs.input_values.to(device))

        # save activations in memory
        extracted_acts = extr.get_activations()
        for layer in extracted_acts.keys():
            for path, act in zip(batch_data["audio_path"], extracted_acts[layer]):
                activations.setdefault(layer, {})
                activations[layer][Path(path).stem] = act

        # clean
        extr.clear()
        del inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()

    results = []
    model_name = Path(model_name_or_path).name
    results_path = Path("results") / model_name / "abx.csv"
    for layer in activations.keys():
        # map path to activations
        def maker(path: str) -> torch.Tensor:
            return activations[layer][Path(path).stem]

        dataset = Dataset.from_item(
            path_items, path, frequency=50, feature_maker=maker, extension=".flac"
        )
        task = Task(dataset, on="#phone", by=["speaker", "next-phone", "prev-phone"])
        score = Score(task, "angular")
        error_rate = score.collapse(levels=[("next-phone", "prev-phone"), "speaker"])
        results.append(
            {
                "model": model_name,
                "condition": "within-speaker",
                "layer": layer,
                "error_rate": error_rate,
                "accuracy": 1 - error_rate,
                "item": path_items,
            }
        )
        # clean
        del dataset, task, score
        gc.collect()

    # save results
    results_path.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name_or_path", type=str, help="Path to the model or model name"
    )
    parser.add_argument("audio_dir", type=str, help="Path to the audio directory")
    parser.add_argument("file_format", type=str, help="Audio file format")
    parser.add_argument("path_items", type=str, help="Path to the items file")
    parser.add_argument(
        "path_audio_flat", type=str, help="Path to the flattened audio directory"
    )
    args = parser.parse_args()
    evaluate_abx(
        args.model_name_or_path,
        audio_dir=args.audio_dir,
        file_format=args.file_format,
        path_items=args.path_items,
        path_audio_flat=args.path_audio_flat,
    )
