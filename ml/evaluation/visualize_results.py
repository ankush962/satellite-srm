import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from ml.datasets.dataloader import create_dataloaders
from ml.models.baseline import BaselineSR
from ml.models.residual_sr import ResidualSR


# ============================================================
# CONFIGURATION
# ============================================================

BASELINE_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_model.pth"
)

RESIDUAL_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_residual_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "visualizations"
)

BATCH_SIZE = 1

NUM_IMAGES = 5


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.backends.mps.is_available():

        return torch.device("mps")

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_model(
    model,
    checkpoint_path,
    device,
):
    """
    Load a trained model checkpoint.
    """

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

        else:

            model.load_state_dict(
                checkpoint
            )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.eval()

    return model


# ============================================================
# CONVERT IMAGE FOR DISPLAY
# ============================================================

def tensor_to_display(
    tensor,
):
    """
    Convert tensor [C, H, W]
    into an image suitable for matplotlib.

    The dataset has 4 channels.
    For visualization we display
    the first three channels as RGB.
    """

    tensor = tensor.detach().cpu()

    tensor = torch.clamp(
        tensor,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Use first 3 channels for RGB visualization
    # --------------------------------------------------------

    image = tensor[:3]

    # Convert:
    # [C, H, W]
    #
    # To:
    # [H, W, C]

    image = image.permute(
        1,
        2,
        0,
    )

    return image.numpy()


# ============================================================
# CREATE VISUALIZATION
# ============================================================

def create_comparison(
    image_index,
    lr,
    bicubic,
    baseline,
    residual,
    hr,
):
    """
    Create and save a five-panel comparison.
    """

    figure, axes = plt.subplots(
        1,
        5,
        figsize=(22, 5),
    )

    images = [

        (
            tensor_to_display(lr),
            "Low Resolution",
        ),

        (
            tensor_to_display(bicubic),
            "Bicubic",
        ),

        (
            tensor_to_display(baseline),
            "Baseline CNN",
        ),

        (
            tensor_to_display(residual),
            "ResidualSR",
        ),

        (
            tensor_to_display(hr),
            "Ground Truth",
        ),

    ]

    for axis, (
        image,
        title,
    ) in zip(
        axes,
        images,
    ):

        axis.imshow(
            image
        )

        axis.set_title(
            title,
            fontsize=12,
        )

        axis.axis(
            "off"
        )

    figure.suptitle(
        f"Satellite Super-Resolution Comparison "
        f"#{image_index}",
        fontsize=16,
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"comparison_{image_index}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved visualization: "
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 60)

    print(
        "GENERATING VISUAL MODEL COMPARISONS"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = get_device()

    print()

    print(
        f"Using device: {device}"
    )

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    print()

    print(
        "Loading test dataset..."
    )

    _, _, test_loader = (
        create_dataloaders(
            batch_size=BATCH_SIZE,
            num_workers=0,
        )
    )

    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    # --------------------------------------------------------
    # LOAD BASELINE MODEL
    # --------------------------------------------------------

    print()

    print(
        "Loading Baseline CNN..."
    )

    baseline_model = (
        BaselineSR()
        .to(device)
    )

    baseline_model = load_model(
        model=baseline_model,
        checkpoint_path=BASELINE_CHECKPOINT,
        device=device,
    )

    print(
        "Baseline CNN loaded successfully!"
    )

    # --------------------------------------------------------
    # LOAD RESIDUAL MODEL
    # --------------------------------------------------------

    print()

    print(
        "Loading ResidualSR..."
    )

    residual_model = (
        ResidualSR()
        .to(device)
    )

    residual_model = load_model(
        model=residual_model,
        checkpoint_path=RESIDUAL_CHECKPOINT,
        device=device,
    )

    print(
        "ResidualSR loaded successfully!"
    )

    # --------------------------------------------------------
    # GENERATE COMPARISONS
    # --------------------------------------------------------

    print()

    print(
        "Generating visual comparisons..."
    )

    print(
        "-" * 60
    )

    with torch.no_grad():

        for image_index, batch in enumerate(
            test_loader
        ):

            if image_index >= NUM_IMAGES:

                break

            lr = batch["lr"].to(
                device
            )

            hr = batch["hr"].to(
                device
            )

            # ------------------------------------------------
            # BICUBIC
            # ------------------------------------------------

            bicubic = F.interpolate(
                lr,
                size=hr.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )

            bicubic = torch.clamp(
                bicubic,
                0.0,
                1.0,
            )

            # ------------------------------------------------
            # BASELINE CNN
            # ------------------------------------------------

            baseline_prediction = (
                baseline_model(lr)
            )

            baseline_prediction = torch.clamp(
                baseline_prediction,
                0.0,
                1.0,
            )

            # ------------------------------------------------
            # RESIDUALSR
            # ------------------------------------------------

            residual_prediction = (
                residual_model(lr)
            )

            residual_prediction = torch.clamp(
                residual_prediction,
                0.0,
                1.0,
            )

            # ------------------------------------------------
            # CREATE VISUALIZATION
            # ------------------------------------------------

            create_comparison(
                image_index=image_index + 1,
                lr=lr[0],
                bicubic=bicubic[0],
                baseline=baseline_prediction[0],
                residual=residual_prediction[0],
                hr=hr[0],
            )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print(
        "VISUAL COMPARISONS GENERATED SUCCESSFULLY"
    )

    print("=" * 60)

    print()

    print(
        f"Saved {NUM_IMAGES} comparisons to:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()