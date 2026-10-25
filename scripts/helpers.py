import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from pathlib import Path

import torch

from models.crossflow import TransformerDIC
from models.efficientnet_encoder_decoder import EffNetEncDec
from models.efficientnet_transformer import EffnetTranformer

MODELS = {
    "model": TransformerDIC,
    "model_1": EffNetEncDec,
    "model_2": EffnetTranformer,
}


def create_model(name, pretrained=True):
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODELS)}")
    return MODELS[name](pretrained=pretrained)


def save_model(model, target_dir, model_name):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / model_name
    print(f"[INFO] Saving model to: {path}")
    torch.save(model.state_dict(), path)


def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Saving CSV to: {path}")


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saving JSON to: {path}")

def get_device() -> torch.device:
    return (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else "cpu"
    )