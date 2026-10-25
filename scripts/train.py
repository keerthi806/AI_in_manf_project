import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import pandas as pd
import torch

from data_prep import get_df_filtered
from dataloader import get_dataloaders
from engine import train
from helpers import MODELS, create_model, save_csv, save_json
from models.crossflow import CharbonnierEndpointLoss
from plotting import plot_loss_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DIC model.")

    parser.add_argument("--model", choices=list(MODELS), default="model")
    parser.add_argument("--data-path", default="AI_mfg_data")
    parser.add_argument("--data-frac", type=float, default=0.3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="mps")

    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--plot", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    df_filtered = get_df_filtered(args.data_path)

    train_dl, test_dl = get_dataloaders(
        df_filtered,
        data_frac=args.data_frac,
        test_size=args.test_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = create_model(args.model, pretrained=True)
    print(f"[INFO] Model: {model.name}")

    loss_func = CharbonnierEndpointLoss()
    optimizer = torch.optim.Adam(
        lr=args.lr,
        params=model.parameters(),
    )

    results = train(
        model=model,
        train_dataloader=train_dl,
        test_dataloader=test_dl,
        optimizer=optimizer,
        loss_fn=loss_func,
        epochs=args.epochs,
        device=args.device,
        save_dir=args.save_dir,
        model_name=f"{model.name}.pth",
    )

    save_csv(pd.DataFrame(results), f"{args.results_dir}/{model.name}.csv")
    save_json(results, f"{args.results_dir}/{model.name}.json")

    if args.plot:
        plot_loss_curves(
            results,
            save_path=f"{args.results_dir}/{model.name}_loss_curves.png",
        )


if __name__ == "__main__":
    main()
