import argparse
import gc
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


def extract(
    model_name_or_path: str,
    audio_dir: str | Path,
    file_format: str = "flac",
    out_dir: str | Path = ACTIVATIONDIR,
    batch_size: int = 16,
):
    """
    Save the activations of every layer of the model for every audio file in audio_dir.

    Args:
        model_name_or_path (str): The name or path of the model (see src/models.py).
        audio_dir (str | Path): The directory containing the audio files (searched recursively).
        file_format (str): The format of the audio files (e.g., "flac").
        out_dir (str | Path): Activations go to <out_dir>/<model name>/<layer>/<file id>.pt.
        batch_size (int): Files per forward pass. Files are sorted by duration, so padding is small
            (median ~0.5% at 16), but the models attend to and normalize over it, so activations differ
            slightly from one file at a time (batch_size=1, exact).
    """
    audio_dir = Path(audio_dir)
    model_dir = Path(out_dir) / Path(model_name_or_path).name

    model, preprocessor, frequency = get_model(model_name_or_path)
    model.eval().to(device)
    annotations = make_annotations(audio_dir, file_format=file_format)
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
                outputs = model(inputs.input_values, inputs.attention_mask)
            else:
                outputs = model(inputs.input_values)

        # save activations to disk, one file per layer and audio file
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
                # drop the frames that only cover batch padding: keep ceil(duration * frame rate),
                # which is at most one frame past the end of the audio (never used by the item files)
                duration = len(signal) / batch_data["audio_sampling_rate"]
                n_frames = min(act.shape[0], math.ceil(duration * frequency))
                # contiguous copy, so only this file's frames are saved (not the whole batch's storage)
                torch.save(
                    act[:n_frames].clone(memory_format=torch.contiguous_format),
                    layer_dir / f"{Path(audio_path).stem}.pt",
                )

        # clean
        extr.clear()
        del inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()

    # what run_abx.py needs to read the activations back
    info = {
        "model": model_name_or_path,
        "frequency": frequency,
        "layers": layers,
        "audio_dir": str(audio_dir),
    }
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
