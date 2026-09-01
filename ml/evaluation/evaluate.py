import torch
import torch.nn as nn

from ml.datasets.dataloader import create_dataloaders
from ml.models.baseline import BaselineSR


DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

CHECKPOINT_PATH = "checkpoints/best_model.pth"


def calculate_psnr(mse):
    """
    Calculate PSNR assuming pixel values are in [0, 1].
    """

    if mse == 0:
        return float("inf")

    return 10 * torch.log10(
        torch.tensor(1.0, device=DEVICE) / mse
    )


def main():

    print("=" * 60)
    print("BASELINE MODEL EVALUATION")
    print("=" * 60)

    print()
    print(f"Using device: {DEVICE}")

    # --------------------------------------------------
    # Load DataLoaders
    # --------------------------------------------------

    print("\nLoading dataset...")

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=8,
        num_workers=0,
    )

    print(f"Test samples: {len(test_loader.dataset)}")

    # --------------------------------------------------
    # Create Model
    # --------------------------------------------------

    model = BaselineSR().to(DEVICE)

    print("\nLoading best model...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    # Support either checkpoint format
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print("Model loaded successfully!")

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    criterion = nn.MSELoss()

    total_loss = 0.0
    total_psnr = 0.0
    total_samples = 0

    print("\nEvaluating test dataset...")
    print("-" * 60)

    with torch.no_grad():

        for batch_index, batch in enumerate(test_loader):

            lr = batch["lr"].to(DEVICE)
            hr = batch["hr"].to(DEVICE)

            prediction = model(lr)

            # Clamp predictions to valid image range
            prediction = torch.clamp(prediction, 0.0, 1.0)

            mse = criterion(prediction, hr)

            psnr = calculate_psnr(mse)

            batch_size = lr.size(0)

            total_loss += mse.item() * batch_size
            total_psnr += psnr.item() * batch_size
            total_samples += batch_size

            if (batch_index + 1) % 10 == 0:
                print(
                    f"Processed batch "
                    f"{batch_index + 1}/{len(test_loader)}"
                )

    average_loss = total_loss / total_samples
    average_psnr = total_psnr / total_samples

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(f"\nTest Samples: {total_samples}")
    print(f"Test MSE Loss: {average_loss:.8f}")
    print(f"Test PSNR: {average_psnr:.4f} dB")

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()