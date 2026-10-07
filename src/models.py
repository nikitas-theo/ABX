from transformers import AutoModel, Wav2Vec2ForCTC, HubertForCTC, WavLMForCTC
from huggingface_hub import hf_hub_download
import torch
from internal_tools.preprocessors import AudioPreprocessor
from internal_tools.models import (
    load_CPC_model,
    load_MelHuBERT_model,
    load_BabyHuBERT_model,
)
from src.config import MODELDIR


def get_model(model_name_or_path: str):
    """Return (model, preprocessor, frame rate of the activations in Hz)."""
    # mirrors internal_tools/tutorials/model_loading_utils.py
    frequency = 50  # 20 ms frames
    match model_name_or_path:
        case (
            "facebook/wav2vec2-base"
            | "facebook/hubert-base-ls960"
            | "microsoft/wavlm-base"
        ):
            model = AutoModel.from_pretrained(model_name_or_path, cache_dir=MODELDIR)
            preprocessor = AudioPreprocessor.for_hf_model(
                model_name_or_path, cache_dir=MODELDIR
            )
        case "facebook/wav2vec2-base-960h":
            model = Wav2Vec2ForCTC.from_pretrained(
                model_name_or_path, cache_dir=MODELDIR
            )
            preprocessor = AudioPreprocessor.for_hf_model(
                model_name_or_path, cache_dir=MODELDIR
            )
        case "facebook/hubert-large-ls960-ft":
            model = HubertForCTC.from_pretrained(model_name_or_path, cache_dir=MODELDIR)
            preprocessor = AudioPreprocessor.for_hf_model(
                model_name_or_path, cache_dir=MODELDIR
            )
        case "patrickvonplaten/wavlm-libri-clean-100h-base":
            model = WavLMForCTC.from_pretrained(model_name_or_path, cache_dir=MODELDIR)
            preprocessor = AudioPreprocessor.for_hf_model(
                model_name_or_path, cache_dir=MODELDIR
            )
        case "MarvinLvn/BabyHuBERT":
            ckpt = hf_hub_download(
                repo_id=model_name_or_path,
                filename="BabyHuBERT.ckpt",
                cache_dir=MODELDIR,
            )
            model = load_BabyHuBERT_model(ckpt)
            preprocessor = AudioPreprocessor.for_hf_model(
                model_name_or_path, cache_dir=MODELDIR
            )
        case "spidr":
            model = torch.hub.load("facebookresearch/spidr", "spidr_base")
            preprocessor = AudioPreprocessor.for_spidr_model()
        case "cpc":
            model = load_CPC_model(MODELDIR / "cpc_checkpoint_106.pt")
            preprocessor = AudioPreprocessor.for_cpc_model()
            frequency = (
                100  # CPC encoder strides 5*4*2*2*2 = 160 samples = 10 ms frames
            )
        case "melhubert":
            model = load_MelHuBERT_model(MODELDIR / "melhubert_960_stage2_20ms.ckpt")
            preprocessor = AudioPreprocessor.for_melhubert_model()
        case _:
            raise ValueError(f"Unsupported model: {model_name_or_path}")
    return model, preprocessor, frequency
