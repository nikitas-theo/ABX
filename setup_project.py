"""Set up everything ABX needs, in order:

1. LibriSpeech audio            -> data/LibriSpeech/<split>/
2. internal_tools (pinned)      -> external/internal_tools/
3. Python environment           (uv sync)
4. Model weights                -> models/

The ZeroSpeech 2021 item files are committed in items/ (from
https://cognitive-ml.fr/downloads/phoneme-discovery/zerospeech2021-triphone.tar.gz).

Safe to re-run: each step skips what is already there.
Uses only the standard library, so it runs with any Python 3.8+ before the environment exists.

Usage:
    python3 setup_project.py
    python3 setup_project.py --splits dev-clean test-clean --models facebook/wav2vec2-base cpc
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LIBRISPEECH = DATA / "LibriSpeech"
EXTERNAL = ROOT / "external"
MODELS = ROOT / "models"

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/{split}.tar.gz"
INTERNAL_TOOLS_REPO = "https://github.com/mdhk/internal_tools.git"
INTERNAL_TOOLS_REV = "11dac9c"
# CPC and MelHuBERT checkpoints come from the internal_tools tutorial models.zip on Google Drive,
# see internal_tools/tutorials/download_tutorial_files.sh
TUTORIAL_MODELS_GDRIVE_ID = "129Fkg_bQpVB_yN-YT5MVK6ulZS_zjhH6"
CHECKPOINTS = {
    "cpc": "cpc_checkpoint_106.pt",
    "melhubert": "melhubert_960_stage2_20ms.ckpt",
}


def step(message):
    print(f"==> {message}", flush=True)


def run(*command):
    """Run a command from the project root, stopping the setup if it fails."""
    subprocess.run(command, cwd=ROOT, check=True)


def download(url, dest):
    """Download url to dest, via a .part file so an interrupted download is never mistaken for a finished one."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"    downloading {url}", flush=True)
    with urllib.request.urlopen(url) as response, open(part, "wb") as f:
        total = int(response.headers.get("Content-Length", 0))
        done, last_shown = 0, -1
        while chunk := response.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            percent = 100 * done // total if total else 0
            if total and percent // 10 != last_shown:
                last_shown = percent // 10
                print(f"    {percent}% of {total / 1e6:.0f} MB", flush=True)
    part.rename(dest)


def extract(archive, dest):
    print(f"    extracting {archive.name}", flush=True)
    with tarfile.open(archive) as tar:
        if hasattr(
            tarfile, "data_filter"
        ):  # Python 3.12+: refuse unsafe paths in the archive
            tar.extractall(dest, filter="data")
        else:
            tar.extractall(dest)


def load_env_file():
    """Read KEY=VALUE lines from .env (e.g. HF_TOKEN) so the download tools pick them up."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def setup_librispeech(splits):
    step(f"LibriSpeech ({', '.join(splits)}) -> {LIBRISPEECH.relative_to(ROOT)}/")
    for split in splits:
        if (LIBRISPEECH / split).is_dir():
            continue
        archive = DATA / f"{split}.tar.gz"
        download(LIBRISPEECH_URL.format(split=split), archive)
        extract(
            archive, DATA
        )  # the archive contains LibriSpeech/<split>/<speaker>/<chapter>/*.flac


def setup_internal_tools():
    target = EXTERNAL / "internal_tools"
    step(f"internal_tools @ {INTERNAL_TOOLS_REV} -> {target.relative_to(ROOT)}/")
    if not target.is_dir():
        run("git", "clone", "--quiet", INTERNAL_TOOLS_REPO, str(target))
    run("git", "-C", str(target), "checkout", "--quiet", INTERNAL_TOOLS_REV)


def setup_environment():
    # installs everything in pyproject.toml, including internal_tools from external/ (editable)
    step("Python environment (uv sync)")
    run("uv", "sync")


def setup_models(models):
    step(f"Models ({', '.join(models)}) -> {MODELS.relative_to(ROOT)}/")
    MODELS.mkdir(exist_ok=True)
    for model in models:
        if model in CHECKPOINTS:
            download_tutorial_checkpoint(CHECKPOINTS[model])
        elif model == "spidr":
            print("    spidr is loaded through torch.hub when it is first used")
        else:  # a HuggingFace model id
            run("uv", "run", "hf", "download", model, "--cache-dir", str(MODELS))


def download_tutorial_checkpoint(filename):
    if (MODELS / filename).exists():
        return
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "models.zip"
        run("uvx", "gdown", TUTORIAL_MODELS_GDRIVE_ID, "-O", str(archive))
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
        # the zip's folder layout is not documented, so search it for the checkpoint by name
        found = list(Path(tmp).rglob(filename))
        if not found:
            sys.exit(f"{filename} not found in the tutorial models.zip")
        shutil.move(str(found[0]), MODELS / filename)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["dev-clean", "dev-other"],
        help="LibriSpeech splits: dev-clean dev-other test-clean test-other",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["facebook/wav2vec2-base"],
        help="HuggingFace model ids, or cpc / melhubert / spidr",
    )
    args = parser.parse_args()

    if shutil.which("uv") is None:
        sys.exit("uv is not installed: https://docs.astral.sh/uv/")
    load_env_file()

    setup_librispeech(args.splits)
    setup_internal_tools()
    setup_environment()
    setup_models(args.models)
    print("Done.")


if __name__ == "__main__":
    main()
