from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from ml.datasets.dataloader import create_dataloaders
from ml.models.baseline import BaselineSR


DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

CHECKPOINT_PATH = "checkpoints/best_model.pth"
OUTPUT_DIR = Path("results/visualizations")

NUM_SAMPLES = 5


def load_model():

    model = BaselineSR().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model


def tensor_to_rgb(image):
    """
    Convert a 4-band tensor to RGB for visualization.

    Dataset band order is assumed to contain:
    B2, B3, B4, B8.

    RGB uses:
    R = B4
    G = B3
    B = B2
    """

    image = image.detach().cpu()

    rgb = torch.stack([
        image[2],  # Red
        image[1],  # Green
        image[0],  # Blue
    ])

    rgb = rgb.permute(1, 2, 0)

    rgb = torch.clamp(rgb, 0.0, 1.0)

    return rgb.numpy()


def main():

    print("=" * 60)
    print("GENERATING VISUAL RESULTS")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"\nUsing device: {DEVICE}")

    print("\nLoading test dataset...")

    _, _, test_loader = create_dataloaders(
        batch_size=1,
        num_workers=0,
    )

    print(f"Test samples: {len(test_loader.dataset)}")

    print("\nLoading model...")

    model = load_model()

    print("Model loaded successfully!")

    print("\nGenerating predictions...")

    saved = 0

    with torch.no_grad():

        for index, batch in enumerate(test_loader):

            if saved >= NUM_SAMPLES:
                break

            lr = batch["lr"].to(DEVICE)
            hr = batch["hr"].to(DEVICE)

            prediction = model(lr)

            prediction = torch.clamp(
                prediction,
                0.0,
                1.0,
            )

            # Upscale LR for fair visual comparison
            lr_upscaled = F.interpolate(
                lr,
                size=hr.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            lr_rgb = tensor_to_rgb(
                lr_upscaled[0]
            )

            prediction_rgb = tensor_to_rgb(
                prediction[0]
            )

            hr_rgb = tensor_to_rgb(
                hr[0]
            )

            fig, axes = plt.subplots(
                1,
                3,
                figsize=(15, 5),
            )

            axes[0].imshow(lr_rgb)
            axes[0].set_title(
                "LR Input (Upscaled)"
            )
            axes[0].axis("off")

            axes[1].imshow(prediction_rgb)
            axes[1].set_title(
                "Model Prediction"
            )
            axes[1].axis("off")

            axes[2].imshow(hr_rgb)
            axes[2].set_title(
                "Ground Truth HR"
            )
            axes[2].axis("off")

            plt.tight_layout()

            output_path = (
                OUTPUT_DIR
                / f"comparison_{saved + 1}.png"
            )

            plt.savefig(
                output_path,
                dpi=150,
                bbox_inches="tight",
            )

            plt.close()

            print(
                f"Saved: {output_path}"
            )

            saved += 1

    print()
    print("=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)

    print(
        f"\nGenerated {saved} comparison images."
    )

    print(
        f"Results saved in: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()