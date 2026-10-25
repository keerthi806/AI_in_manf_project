"""Training and testing loops."""

import time
from pathlib import Path

import torch
from tqdm.auto import tqdm


def save_model(model, target_dir, model_name):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / model_name
    print(f"Saving model to: {path}")
    torch.save(model.state_dict(), path)


def train_step(epoch, model, dataloader, loss_fn, optimizer, device, disable_progress_bar=False):
    model.train()
    train_loss = 0.0

    progress_bar = tqdm(
        enumerate(dataloader),
        desc=f"Training Epoch {epoch}",
        total=len(dataloader),
        disable=disable_progress_bar,
    )

    for batch, (X1, X2, y) in progress_bar:
        X1, X2, y = X1.to(device), X2.to(device), y.to(device)

        y_pred = model(X1, X2)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix({"train_loss": train_loss / (batch + 1)})

    return train_loss / len(dataloader)


def test_step(epoch, model, dataloader, loss_fn, device, disable_progress_bar=False):
    model.eval()
    test_loss = 0.0

    progress_bar = tqdm(
        enumerate(dataloader),
        desc=f"Testing Epoch {epoch}",
        total=len(dataloader),
        disable=disable_progress_bar,
    )

    with torch.no_grad():
        for batch, (X1, X2, y) in progress_bar:
            X1, X2, y = X1.to(device), X2.to(device), y.to(device)

            y_pred = model(X1, X2)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            progress_bar.set_postfix({"test_loss": test_loss / (batch + 1)})

    return test_loss / len(dataloader)


def train(model, train_dataloader, test_dataloader, optimizer, loss_fn,
          epochs, device, disable_progress_bar=False, scheduler=None,
          save_dir="models", model_name="AI_mfg_model.pth"):
    results = {
        "train_loss": [],
        "test_loss": [],
        "train_epoch_time": [],
        "test_epoch_time": [],
    }

    model.to(device)
    best_test_loss = float("inf")

    for epoch in tqdm(range(epochs), disable=disable_progress_bar):
        print(f"Using LR: {optimizer.param_groups[0]['lr']}")

        start = time.time()
        train_loss = train_step(
            epoch, model, train_dataloader, loss_fn, optimizer, device,
            disable_progress_bar,
        )
        train_epoch_time = time.time() - start

        start = time.time()
        test_loss = test_step(
            epoch, model, test_dataloader, loss_fn, device,
            disable_progress_bar,
        )
        test_epoch_time = time.time() - start

        if scheduler:
            scheduler.step()

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"train_epoch_time: {train_epoch_time:.4f} | "
            f"test_epoch_time: {test_epoch_time:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["train_epoch_time"].append(train_epoch_time)
        results["test_epoch_time"].append(test_epoch_time)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_model(model, save_dir, model_name)

    return results
