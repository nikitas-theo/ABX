import torch
from huggingface_hub import hf_hub_download
from internal_tools.models import (
    load_BabyHuBERT_model,
    load_CPC_model,
    load_MelHuBERT_model,
)
from internal_tools.preprocessors import AudioPreprocessor
from transformers import AutoModel

from src.config import MODELDIR


def get_model(model_name_or_path: str):
    """Return (model, preprocessor, frame rate of the activations in Hz).

    Mirrors internal_tools/tutorials/model_loading_utils.py.
    """
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
            return model, preprocessor, 50
        case "MarvinLvn/BabyHuBERT":
            ckpt = hf_hub_download(
                model_name_or_path, "BabyHuBERT.ckpt", cache_dir=MODELDIR
            )
            # no preprocessor config on the hub: internal_tools falls back to wav2vec2-base's
            preprocessor = AudioPreprocessor.for_hf_model(
                model_name_or_path, cache_dir=MODELDIR
            )
            return load_BabyHuBERT_model(ckpt), preprocessor, 50
        case "spidr":
            model = torch.hub.load("facebookresearch/spidr", "spidr_base")
            # the channels-last CNN path bypasses conv_layers[-1].conv, where internal_tools hooks "CNN"
            model.feature_extractor.channels_last = False
            return model, AudioPreprocessor.for_spidr_model(), 50
        case "melhubert":
            model = load_MelHuBERT_model(MODELDIR / "melhubert_960_stage2_20ms.ckpt")
            return model, AudioPreprocessor.for_melhubert_model(), 50
        case "cpc":
            # 10 ms frames: the encoder strides multiply to 160 samples
            model = load_CPC_model(MODELDIR / "cpc_checkpoint_106.pt")
            return model, AudioPreprocessor.for_cpc_model(), 100
        case _:
            raise ValueError(f"Unsupported model: {model_name_or_path}")
