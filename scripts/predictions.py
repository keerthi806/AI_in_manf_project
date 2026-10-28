import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
import json

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy import io

from data_prep import get_df_filtered
from plotting import plot_loss_curves, visualize_dic

from models.crossflow import TransformerDIC
from models.efficientnet_encoder_decoder import EffNetEncDec
from models.efficientnet_transformer import EffnetTranformer

MODEL_ARCS = (TransformerDIC, EffNetEncDec, EffnetTranformer)

# Checkpoint paths sorted alphabetically, zipped with MODEL_ARCS (as in the
# original predict.ipynb cells 14-17).
CHECKPOINT_PATHS = sorted(
    p for p in Path("../models/checkpoints").iterdir() if p.suffix == ".pth"
)

RESULTS_DIR = Path("results")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict(
    model: torch.nn.Module,
    img1,
    img2,
    accelerator: bool = True,
):
    from torchvision import transforms

    to_tensor = transforms.ToTensor()

    if not isinstance(img1, torch.Tensor):
        img1 = to_tensor(img1)
    if not isinstance(img2, torch.Tensor):
        img2 = to_tensor(img2)

    if img1.ndim != 3 or img2.ndim != 3:
        raise ValueError(
            f"Expected images with shape [C, H, W]; "
            f"got {tuple(img1.shape)} and {tuple(img2.shape)}"
        )

    if img1.shape != img2.shape:
        raise ValueError(
            f"Input image shapes differ: "
            f"{tuple(img1.shape)} vs {tuple(img2.shape)}"
        )

    device = get_device() if accelerator else torch.device("cpu")

    model = model.to(device)
    img1 = img1.unsqueeze(0).to(device=device, dtype=torch.float32)
    img2 = img2.unsqueeze(0).to(device=device, dtype=torch.float32)

    model.eval()

    with torch.inference_mode():
        disp_field = model(img1, img2)

    if disp_field.ndim != 4 or disp_field.shape[0] != 1:
        raise ValueError(
            f"Expected output [1, 2, H, W], got {tuple(disp_field.shape)}"
        )

    return disp_field[0].detach().cpu()


def create_sample(paths_df: pd.DataFrame, seed: int = 42):
    random.seed(seed)
    index = random.randint(0, len(paths_df))
    img1_path, img2_path, disp_field_path = paths_df.iloc[index]
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    mat_data = io.loadmat(disp_field_path)

    if "Disp_field_1" in mat_data:
        disp_field = mat_data["Disp_field_1"]
    elif "Disp_field_2" in mat_data:
        disp_field = mat_data["Disp_field_2"]
    else:
        raise KeyError(
            "Neither Disp_field_1 nor Disp_field_2 found in MAT file."
        )

    return img1, img2, disp_field


def compare_displacement(gt, pred):
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().numpy()

    gt = np.asarray(gt)
    pred = np.asarray(pred)

    if pred.ndim == 4:
        if pred.shape[0] != 1:
            raise ValueError("Expected prediction batch size 1.")
        pred = pred[0]

    if pred.ndim == 3 and pred.shape[0] == 2:
        pred = pred.transpose(1, 2, 0)

    if gt.shape != pred.shape or gt.ndim != 3 or gt.shape[-1] != 2:
        raise ValueError(
            f"Expected matching (H, W, 2) fields; got gt={gt.shape}, "
            f"pred={pred.shape}"
        )

    error = pred - gt
    epe = np.linalg.norm(error, axis=-1)

    metrics = {
        "Mean EPE (pixels)": float(np.mean(epe)),
        "Median EPE (pixels)": float(np.median(epe)),
        "EPE RMSE (pixels)": float(np.sqrt(np.mean(epe ** 2))),
        "Mean absolute ux error": float(np.mean(np.abs(error[..., 0]))),
        "Mean absolute uy error": float(np.mean(np.abs(error[..., 1]))),
        "Maximum EPE (pixels)": float(np.max(epe)),
    }

    print("\nDisplacement Comparison")
    print("-" * 35)
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    return metrics, epe


def plot_displacement_quiver(gt, pred, stride=12, arrow_scale=0.5):
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    gt = np.asarray(gt)
    pred = np.asarray(pred)

    if pred.ndim == 4:
        if pred.shape[0] != 1:
            raise ValueError("Expected prediction batch size 1.")
        pred = pred[0]

    if pred.ndim == 3 and pred.shape[0] == 2:
        pred = pred.transpose(1, 2, 0)

    if gt.ndim != 3 or gt.shape[-1] != 2 or pred.shape != gt.shape:
        raise ValueError(
            f"Expected matching fields of shape (H, W, 2); "
            f"got gt={gt.shape}, pred={pred.shape}"
        )

    H, W, _ = gt.shape
    y, x = np.mgrid[0:H:stride, 0:W:stride]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    fields = [gt, pred]
    titles = ["Ground Truth Displacement", "Predicted Displacement"]

    for ax, field, title in zip(axes, fields, titles):
        ux = field[::stride, ::stride, 0]
        uy = field[::stride, ::stride, 1]

        magnitude = np.sqrt(ux ** 2 + uy ** 2)

        ax.quiver(
            x, y, ux, -uy,
            magnitude,
            cmap="turbo",
            angles="xy",
            scale_units="xy",
            scale=arrow_scale,
            pivot="tail",
            width=0.003,
            headwidth=4,
            headlength=5,
        )

        ax.set_title(title)
        ax.set_xlim(0, W - 1)
        ax.set_ylim(H - 1, 0)
        ax.set_aspect("equal")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

    plt.tight_layout()
    plt.show()


def plot_loss_curves_for_model(model_name: str):
    results_path = RESULTS_DIR / model_name / f"{model_name}.json"
    if not results_path.exists():
        print(f"[WARN] No results file found at {results_path}; skipping loss curve.")
        return

    with open(results_path) as f:
        results = json.load(f)

    plot_loss_curves(
        results,
        model_name=model_name,
    )


def main():
    df = get_df_filtered(data_path="../AI_mfg_data")
    print(f"[INFO] Loaded {len(df)} samples from AI_mfg_data")

    img1, img2, disp_field = create_sample(df)
    print("[INFO] Visualizing the sampled pair")
    visualize_dic(
        initial=img1,
        deformed=img2,
        displacement=disp_field,
        title="Sampled DIC Pair and Ground Truth Displacement",
    )

    models = [model() for model in MODEL_ARCS]

    for model, checkpoint in zip(models, CHECKPOINT_PATHS):
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)

    for model in models:
        print("\n" + "=" * 70)
        print(f"Model: {model.name}")
        print("=" * 70)

        pred = predict(
            model=model,
            img1=img1,
            img2=img2,
            accelerator=False,
        )

        compare_displacement(gt=disp_field, pred=pred)
        plot_displacement_quiver(
            gt=disp_field,
            pred=pred,
            stride=12,
            arrow_scale=0.4,
        )

        pred_np = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
        print(f"\nModel: {model.name}")
        print("Prediction shape:", pred_np.shape)
        print("Prediction range:", pred_np.min(), pred_np.max())
        print("Prediction mean:", pred_np.mean())
        print("Prediction std:", pred_np.std())

    print("\n" + "=" * 70)
    print("Training Loss Curves")
    print("=" * 70)
    for model in models:
        plot_loss_curves_for_model(model.name)

    print("\n[INFO] predictions.py complete.")


if __name__ == "__main__":
    main()
