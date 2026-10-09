"""Prepare the audio and item files for each ABX benchmark.

zero_shot  ZeroSpeech 2021 triphone ABX on LibriSpeech (openslr.org/12, fastabx item files)
    audio  -> data/LibriSpeech/<split>/
    items  -> items/zerospeech2021-triphone/item/triphone-<split>.item
    sample -> items/zerospeech2021-triphone/sample/triphone-<split>-sample.item: 10 speakers
              (5 F, 5 M) x 25 recordings, ~10% of the split, so every layer can be scored quickly

prosodic   Prosodic ABX, English lexical stress (https://arxiv.org/abs/2604.02102), from the
           HaitongSUN/prosody-abx dataset: stress (recorded), stress_syn (Google TTS), stress_kokoro
    audio  -> data/prosodic/<set>/<id>.wav, one word per file, 16 kHz mono
    items  -> items/prosodic/<set>.csv

Safe to re-run: existing files are skipped. Run after setup_project.py.

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

load_dotenv()  # HF_TOKEN

SAMPLING_RATE = 16000

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
        # via a .part file, so an interrupted download is not mistaken for a finished one
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
        archive = DATADIR / "zerospeech2021-triphone.tar.gz"
        download_and_extract(ZEROSPEECH_ITEMS_URL, archive, ZEROSPEECH_ITEMS)
    for split in splits:
        print(f"==> zero_shot: LibriSpeech {split}")
        if not (DATADIR / "LibriSpeech" / split).is_dir():
            archive = DATADIR / f"{split}.tar.gz"
            download_and_extract(LIBRISPEECH_URL.format(split=split), archive, DATADIR)
        make_zerospeech_sample(split)


SAMPLE_SPEAKERS_PER_SEX = 5
SAMPLE_FILES_PER_SPEAKER = 25
SAMPLE_SEED = 0


def make_zerospeech_sample(split):
    """The ZeroSpeech item file restricted to a few speakers and recordings."""
    sample_path = ZEROSPEECH_ITEMS / "sample" / f"triphone-{split}-sample.item"
    if sample_path.exists():
        return
    # SPEAKERS.TXT lines: "ID | SEX | SUBSET | MINUTES | NAME"
    speakers = (
        pl.read_csv(
            DATADIR / "LibriSpeech" / "SPEAKERS.TXT",
            separator="|",
            comment_prefix=";",
            has_header=False,
            new_columns=["id", "sex", "subset", "minutes", "name"],
            infer_schema=False,
            truncate_ragged_lines=True,
        )
        .with_columns(pl.all().str.strip_chars())
        .filter(pl.col("subset") == split)
    )
    selected = [
        speaker
        for sex in ["F", "M"]
        for speaker in speakers.filter(pl.col("sex") == sex)["id"]
        .sample(SAMPLE_SPEAKERS_PER_SEX, seed=SAMPLE_SEED)
        .sort()
        .to_list()
    ]
    items = pl.read_csv(
        ZEROSPEECH_ITEMS / "item" / f"triphone-{split}.item",
        separator=" ",
        infer_schema=False,
    ).filter(pl.col("speaker").is_in(selected))
    files = [
        f
        for speaker in selected
        for f in items.filter(pl.col("speaker") == speaker)["#file"]
        .unique()
        .sort()
        .sample(SAMPLE_FILES_PER_SPEAKER, seed=SAMPLE_SEED)
        .to_list()
    ]
    items = items.filter(pl.col("#file").is_in(files))
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    items.write_csv(sample_path, separator=" ")
    print(
        f"    sample: {len(items)} triphones, {len(files)} recordings, speakers {selected} -> {sample_path}"
    )


PROSODIC_HF_DATASET = "HaitongSUN/prosody-abx"
PROSODIC_SETS = {
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
            if (
                "target_onset" in row
            ):  # recorded speech: cut the word out of its sentence
                audio = audio[
                    round(row["target_onset"] * sr) : round(row["target_offset"] * sr)
                ]
            if sr != SAMPLING_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
            sf.write(
                audio_dir / f"{row['id']}.wav", audio.astype(np.float32), SAMPLING_RATE
            )
            # the prosodic-abx repo's item columns: the word is phone_sequence,
            # the stress pattern accent_pattern
            items.append(
                {
                    "#file": row["id"],
                    "onset": 0.0,
                    "offset": len(audio) / SAMPLING_RATE,
                    "phone_sequence": row["target"],
                    "accent_pattern": row["label"],
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
