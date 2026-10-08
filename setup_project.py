"""Set up everything ABX needs, in order:

1. internal_tools (pinned)      -> external/internal_tools/
2. Python environment           (uv sync)
3. Model weights                -> models/

Then prepare the ABX benchmarks' audio and item files with src/prepare_tasks.py.

Usage:
    python3 setup_project.py
    python3 setup_project.py --models facebook/wav2vec2-base cpc
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXTERNAL = ROOT / "external"
MODELS = ROOT / "models"
INTERNAL_TOOLS_REPO = "https://github.com/mdhk/internal_tools.git"
INTERNAL_TOOLS_REV = "11dac9c"

# CPC and MelHuBERT checkpoints come from the internal_tools tutorial models.zip on Google Drive
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
            # .env holds HF_TOKEN (optional) for authenticated downloads; uv errors if the file is missing
            env_file = ["--env-file", ".env"] if (ROOT / ".env").exists() else []
            run(
                "uv",
                "run",
                *env_file,
                "hf",
                "download",
                model,
                "--cache-dir",
                str(MODELS),
            )


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
        "--models",
        nargs="+",
        default=["facebook/wav2vec2-base"],
        help="HuggingFace model ids, or cpc / melhubert / spidr",
    )
    args = parser.parse_args()

    if shutil.which("uv") is None:
        sys.exit("uv is not installed: https://docs.astral.sh/uv/")

    setup_internal_tools()
    setup_environment()
    setup_models(args.models)
    print("Done.")


if __name__ == "__main__":
    main()
