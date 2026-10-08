"""Prepare the audio and item files for each ABX benchmark.

zero_shot  ZeroSpeech 2021 triphone ABX on LibriSpeech
    audio  -> data/LibriSpeech/<split>/<speaker>/<chapter>/*.flac
    items  -> items/zerospeech2021-triphone/item/triphone-<split>.item
    from https://www.openslr.org/12 and https://docs.cognitive-ml.fr/fastabx/items.html

prosodic   Prosodic ABX, English lexical stress (https://arxiv.org/abs/2604.02102)
    audio  -> data/prosodic/<set>/<id>.wav, one target word per file, 16 kHz mono
    items  -> items/prosodic/<set>.csv, built from the dataset's word labels (it ships no item files)
    sets: stress (natural speech, words cut out of sentences), stress_syn (Google TTS),
          stress_kokoro (Kokoro TTS); from https://huggingface.co/datasets/HaitongSUN/prosody-abx

Safe to re-run: each step skips what is already there. Run after setup_project.py.

Usage:
    python -m src.prepare_tasks zero_shot prosodic
    python -m src.prepare_tasks zero_shot --splits dev-clean
"""

import argparse
import io
import tarfile
import urllib.request

import librosa
import numpy as np
import polars as pl
import soundfile as sf
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from src.config import DATADIR, ITEMDIR

load_dotenv()  # HF_TOKEN from .env, for authenticated downloads

SAMPLING_RATE = 16000  # what all the models expect


# --- zero_shot: ZeroSpeech 2021 triphone ABX on LibriSpeech ---

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/{split}.tar.gz"
ZEROSPEECH_ITEMS_URL = (
    "https://cognitive-ml.fr/downloads/phoneme-discovery/zerospeech2021-triphone.tar.gz"
)
ZEROSPEECH_ITEMS = ITEMDIR / "zerospeech2021-triphone"


def download_and_extract(url, archive, dest):
    """Download a .tar.gz (unless already there) and extract it into dest."""
    if not archive.exists():
        print(f"    downloading {url}")
        archive.parent.mkdir(parents=True, exist_ok=True)
        # via a .part file, so an interrupted download is never mistaken for a finished one
        part = archive.with_name(archive.name + ".part")
        with urllib.request.urlopen(url) as response, open(part, "wb") as f:
            total = int(response.headers.get("Content-Length", 0))
            done, last_shown = 0, -1
            while chunk := response.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and 10 * done // total != last_shown:
                    last_shown = 10 * done // total
                    print(
                        f"    {100 * done // total}% of {total / 1e6:.0f} MB",
                        flush=True,
                    )
        part.rename(archive)
    print(f"    extracting {archive.name}")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")


def prepare_zero_shot(splits):
    print("==> zero_shot: ZeroSpeech 2021 triphone items")
    if not (ZEROSPEECH_ITEMS / "item").is_dir():
        # the archive contains item/triphone-<split>.item for dev/test clean/other
        archive = DATADIR / "zerospeech2021-triphone.tar.gz"
        download_and_extract(ZEROSPEECH_ITEMS_URL, archive, ZEROSPEECH_ITEMS)
    for split in splits:
        print(f"==> zero_shot: LibriSpeech {split}")
        if not (DATADIR / "LibriSpeech" / split).is_dir():
            # the archive contains LibriSpeech/<split>/<speaker>/<chapter>/*.flac
            archive = DATADIR / f"{split}.tar.gz"
            download_and_extract(LIBRISPEECH_URL.format(split=split), archive, DATADIR)


# --- prosodic: Prosodic ABX, English lexical stress ---

PROSODIC_HF_DATASET = "HaitongSUN/prosody-abx"
PROSODIC_SETS = {
    # set name: parquet file in the HF dataset
    "stress": "english_stress/english_stress.parquet",
    "stress_syn": "english_stress_syn/english_stress_syn.parquet",
    "stress_kokoro": "english_stress_kokoro/english_stress_kokoro.parquet",
}


def prepare_prosodic(sets):
    for name in sets:
        print(f"==> prosodic: {name}")
        audio_dir = DATADIR / "prosodic" / name
        item_path = ITEMDIR / "prosodic" / f"{name}.csv"
        if item_path.exists():
            continue
        parquet = hf_hub_download(
            PROSODIC_HF_DATASET,
            PROSODIC_SETS[name],
            repo_type="dataset",
            cache_dir=DATADIR / "hf",
        )
        rows = pl.read_parquet(parquet).sort("id").to_dicts()

        audio_dir.mkdir(parents=True, exist_ok=True)
        items = []
        for row in rows:
            audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if "target_onset" in row:
                # natural speech: cut the target word out of the carrier sentence
                audio = audio[
                    round(row["target_onset"] * sr) : round(row["target_offset"] * sr)
                ]
            if sr != SAMPLING_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
            sf.write(
                audio_dir / f"{row['id']}.wav", audio.astype(np.float32), SAMPLING_RATE
            )
            # same columns as the prosodic-abx repo's item files; the ABX compares the whole word
            items.append(
                {
                    "#file": row["id"],
                    "onset": 0.0,
                    "offset": len(audio) / SAMPLING_RATE,
                    "phone_sequence": row[
                        "target"
                    ],  # the word: BY, so only stress differs
                    "accent_pattern": row["label"],  # stress pattern: ON
                    "speaker": row["speaker"],
                    "lexical_category": row["lexical_category"],
                }
            )
        item_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(items).write_csv(item_path)
        print(
            f"    {len(items)} words from {len({i['speaker'] for i in items})} speakers -> {item_path}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("benchmarks", nargs="+", choices=["zero_shot", "prosodic"])
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["dev-clean", "dev-other"],
        help="zero_shot: LibriSpeech splits",
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        default=list(PROSODIC_SETS),
        choices=list(PROSODIC_SETS),
        help="prosodic: sets",
    )
    args = parser.parse_args()

    if "zero_shot" in args.benchmarks:
        prepare_zero_shot(args.splits)
    if "prosodic" in args.benchmarks:
        prepare_prosodic(args.sets)


if __name__ == "__main__":
    main()
