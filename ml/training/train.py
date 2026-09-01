import sys
from pathlib import Path
import time
import json

import torch
import torch.nn as nn
from torch.optim import Adam


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from ml.datasets.dataloader import create_dataloaders
from ml.models.baseline import BaselineSR


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 8
NUM_WORKERS = 0

LEARNING_RATE = 1e-3
EPOCHS = 20

MODEL_NAME = "Baseline CNN"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results" / "training"

BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "best_baseline_model.pth"
)

LATEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "latest_baseline_model.pth"
)

HISTORY_PATH = (
    RESULTS_DIR / "baseline_training_history.json"
)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.backends.mps.is_available():
        print("Using Apple Silicon MPS GPU")
        return torch.device("mps")

    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")

    print("Using CPU")
    return torch.device("cpu")


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    total_loss = 0.0

    for batch_index, batch in enumerate(loader):

        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)

        # Forward pass
        prediction = model(lr)

        # Loss
        loss = criterion(prediction, hr)

        # Clear gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update model
        optimizer.step()

        total_loss += loss.item()

        if (batch_index + 1) % 20 == 0:

            print(
                f"Batch {batch_index + 1}/{len(loader)} "
                f"| Loss: {loss.item():.6f}"
            )

    return total_loss / len(loader)


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for batch in loader:

            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            prediction = model(lr)

            loss = criterion(
                prediction,
                hr,
            )

            total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    train_loss,
    val_loss,
):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_name": MODEL_NAME,
            "dataset_split": "date_based",
            "train_samples": 1661,
            "validation_samples": 446,
            "test_samples": 546,
        },
        path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 65)
    print("BASELINE CNN SUPER-RESOLUTION TRAINING")
    print("=" * 65)
    print()

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = get_device()

    print()

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print("Loading dataset using date-based split...")

    train_loader, val_loader, test_loader = (
        create_dataloaders(
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )
    )

    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)
    test_samples = len(test_loader.dataset)

    print()
    print(f"Train samples: {train_samples}")
    print(f"Validation samples: {val_samples}")
    print(f"Test samples: {test_samples}")

    print()
    print("Dataset split: DATE-BASED")
    print("No acquisition date overlaps between splits.")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()

    model = BaselineSR().to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"{MODEL_NAME} created.")
    print(f"Trainable parameters: {parameters:,}")

    # --------------------------------------------------------
    # LOSS AND OPTIMIZER
    # --------------------------------------------------------

    criterion = nn.MSELoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # DIRECTORIES
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------

    history = {
        "model_name": MODEL_NAME,
        "dataset_split": "date_based",
        "train_samples": train_samples,
        "validation_samples": val_samples,
        "test_samples": test_samples,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": [],
    }

    best_val_loss = float("inf")

    # --------------------------------------------------------
    # TRAINING START
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("STARTING TRAINING")
    print("=" * 65)

    for epoch in range(EPOCHS):

        print()
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 65)

        start_time = time.time()

        # Training
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        # Validation
        val_loss = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_time = time.time() - start_time

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        print()
        print(f"Epoch {epoch + 1} Results")
        print(f"Train Loss: {train_loss:.8f}")
        print(f"Validation Loss: {val_loss:.8f}")
        print(f"Time: {epoch_time:.2f} seconds")

        # Save history
        history["epochs"].append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "time_seconds": epoch_time,
            }
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_checkpoint(
                path=BEST_CHECKPOINT_PATH,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                train_loss=train_loss,
                val_loss=val_loss,
            )

            print()
            print("✓ New best Baseline CNN model saved!")
            print(f"  Path: {BEST_CHECKPOINT_PATH}")

        # ----------------------------------------------------
        # SAVE LATEST MODEL
        # ----------------------------------------------------

        save_checkpoint(
            path=LATEST_CHECKPOINT_PATH,
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            train_loss=train_loss,
            val_loss=val_loss,
        )

        # Save training history after every epoch
        with open(HISTORY_PATH, "w") as file:

            json.dump(
                history,
                file,
                indent=4,
            )

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print()
    print("=" * 65)
    print("BASELINE CNN TRAINING COMPLETE!")
    print("=" * 65)

    print()
    print(f"Best Validation Loss: {best_val_loss:.8f}")

    print()
    print("Best model saved at:")
    print(BEST_CHECKPOINT_PATH)

    print()
    print("Training history saved at:")
    print(HISTORY_PATH)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

