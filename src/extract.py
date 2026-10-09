import argparse
import json
import math
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import tqdm
from dotenv import load_dotenv
from internal_tools.dataset_utils import AnnotatedAudioDataset, aadl_collate_fn
from internal_tools.extractors import AudioModelExtractor
from torch.utils.data import DataLoader

from src.config import ACTIVATIONDIR
from src.models import get_model

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"


def make_annotations(
    audio_dir: str | Path, file_format: str = "flac", file_ids: set | None = None
) -> pd.DataFrame:
    """One annotation spanning each whole audio file, as AnnotatedAudioDataset expects."""
    audio_dir = Path(audio_dir)
    rows = []
    for f in sorted(audio_dir.glob(f"**/*.{file_format}")):
        if file_ids is not None and f.stem not in file_ids:
            continue
        rows.append(
            {
                "file_id": str(f.relative_to(audio_dir).with_suffix("")),
                "start_time": 0.0,
                "end_time": sf.info(f).duration,
            }
        )
    return pd.DataFrame(rows)


def extract(
    model_name_or_path: str,
    audio_dir: str | Path,
    file_format: str = "flac",
    out_dir: str | Path = ACTIVATIONDIR,
    batch_size: int = 16,
    file_ids: set | None = None,
):
    """
    Save every layer's activations for the audio files in audio_dir (searched recursively)
    to <out_dir>/<model name>/<layer>/<file id>.pt, plus info.json for run_abx.py.

    Args:
        model_name_or_path (str): A model accepted by src/models.py.
        file_ids (set | None): Only these files (names without extension); all if None.
        batch_size (int): Files per forward pass. Files are sorted by duration, so padding is
            small, but it still changes the activations slightly; 1 is exact.
    """
    audio_dir = Path(audio_dir)
    model_dir = Path(out_dir) / Path(model_name_or_path).name

    model, preprocessor, frequency = get_model(model_name_or_path)
    model.eval().to(device)
    annotations = make_annotations(
        audio_dir, file_format=file_format, file_ids=file_ids
    )
    dataset = AnnotatedAudioDataset(annotations, audio_dir, file_format=file_format)
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=aadl_collate_fn)
    extr = AudioModelExtractor(model)
    layers = None

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

        with torch.no_grad():
            if model_name_or_path == "melhubert":
                model(inputs.input_values, inputs.attention_mask)
            else:
                model(inputs.input_values)

        extracted_acts = extr.get_activations()
        layers = list(extracted_acts.keys())
        for layer in layers:
            layer_dir = model_dir / layer
            layer_dir.mkdir(parents=True, exist_ok=True)
            for audio_path, signal, act in zip(
                batch_data["audio_path"],
                batch_data["audio_signal"],
                extracted_acts[layer],
            ):
                # drop the frames that only cover batch padding
                duration = len(signal) / batch_data["audio_sampling_rate"]
                n_frames = min(act.shape[0], math.ceil(duration * frequency))
                # a copy, so the file holds only these frames, not the whole batch's storage
                torch.save(
                    act[:n_frames].clone(memory_format=torch.contiguous_format),
                    layer_dir / f"{Path(audio_path).stem}.pt",
                )

        extr.clear()

    info = {"model": model_name_or_path, "frequency": frequency, "layers": layers}
    (model_dir / "info.json").write_text(json.dumps(info, indent=2))
    print(f"Saved activations to {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name_or_path", type=str, help="Path to the model or model name"
    )
    parser.add_argument("audio_dir", type=str, help="Path to the audio directory")
    parser.add_argument(
        "--file_format", type=str, default="flac", help="Audio file format"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=ACTIVATIONDIR,
        help="Where to save the activations",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Files per forward pass"
    )
    args = parser.parse_args()
    extract(
        args.model_name_or_path,
        audio_dir=args.audio_dir,
        file_format=args.file_format,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
    )
