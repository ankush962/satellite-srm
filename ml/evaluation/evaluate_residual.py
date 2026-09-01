import math
import json
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

from ml.datasets.dataloader import create_dataloaders
from ml.models.residual_sr import ResidualSR


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = "checkpoints/best_residual_model.pth"

BATCH_SIZE = 8

OUTPUT_DIR = Path("results/metrics")


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
# PSNR
# ============================================================

def calculate_psnr(mse):
    """
    Calculate PSNR assuming image values are in [0, 1].
    """

    if mse == 0:
        return float("inf")

    return 10 * math.log10(1.0 / mse)


# ============================================================
# SSIM
# ============================================================

def calculate_ssim(prediction, target):
    """
    Calculate average SSIM for a batch.

    prediction: Tensor [B, C, H, W]
    target: Tensor [B, C, H, W]
    """

    prediction = prediction.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    scores = []

    for pred_image, target_image in zip(
        prediction,
        target,
    ):

        # Convert:
        # [C, H, W]
        #
        # To:
        # [H, W, C]

        pred_image = pred_image.transpose(
            1,
            2,
            0,
        )

        target_image = target_image.transpose(
            1,
            2,
            0,
        )

        score = ssim(
            target_image,
            pred_image,
            channel_axis=2,
            data_range=1.0,
        )

        scores.append(
            float(score)
        )

    return sum(scores) / len(scores)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    improvements,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    json_path = (
        OUTPUT_DIR
        / "residual_comparison_results.json"
    )

    json_data = {
        "results": results,
        "improvement_over_bicubic": improvements,
    }

    with open(
        json_path,
        "w",
    ) as file:

        json.dump(
            json_data,
            file,
            indent=4,
        )

    print(
        f"Saved JSON: {json_path}"
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / "residual_comparison_results.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "mse",
                "psnr_db",
                "ssim",
            ],
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print(
        f"Saved CSV: {csv_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("RESIDUALSR vs BICUBIC EVALUATION")
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
    # LOAD DATASET
    # --------------------------------------------------------

    print()
    print("Loading test dataset...")

    _, _, test_loader = (
        create_dataloaders(
            batch_size=BATCH_SIZE
        )
    )

    print(
        f"Test batches: "
        f"{len(test_loader)}"
    )

    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print()
    print("Loading ResidualSR model...")

    model = ResidualSR().to(
        device
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

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

    model.eval()

    print(
        "ResidualSR model loaded successfully!"
    )

    # --------------------------------------------------------
    # METRIC TOTALS
    # --------------------------------------------------------

    residual_mse_total = 0.0

    bicubic_mse_total = 0.0

    residual_ssim_total = 0.0

    bicubic_ssim_total = 0.0

    total_batches = 0

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print()
    print("Evaluating...")
    print("-" * 60)

    with torch.no_grad():

        for batch_index, batch in enumerate(
            test_loader
        ):

            lr = batch["lr"].to(
                device
            )

            hr = batch["hr"].to(
                device
            )

            # ------------------------------------------------
            # RESIDUALSR PREDICTION
            # ------------------------------------------------

            prediction = model(
                lr
            )

            prediction = torch.clamp(
                prediction,
                0.0,
                1.0,
            )

            # ------------------------------------------------
            # BICUBIC BASELINE
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
            # MSE
            # ------------------------------------------------

            residual_mse = (
                F.mse_loss(
                    prediction,
                    hr,
                ).item()
            )

            bicubic_mse = (
                F.mse_loss(
                    bicubic,
                    hr,
                ).item()
            )

            # ------------------------------------------------
            # SSIM
            # ------------------------------------------------

            residual_ssim = (
                calculate_ssim(
                    prediction,
                    hr,
                )
            )

            bicubic_ssim = (
                calculate_ssim(
                    bicubic,
                    hr,
                )
            )

            # ------------------------------------------------
            # ACCUMULATE
            # ------------------------------------------------

            residual_mse_total += (
                residual_mse
            )

            bicubic_mse_total += (
                bicubic_mse
            )

            residual_ssim_total += (
                residual_ssim
            )

            bicubic_ssim_total += (
                bicubic_ssim
            )

            total_batches += 1

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if (
                (batch_index + 1) % 10 == 0
            ):

                print(
                    f"Processed batch "
                    f"{batch_index + 1}/"
                    f"{len(test_loader)}"
                )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    residual_mse = (
        residual_mse_total
        / total_batches
    )

    bicubic_mse = (
        bicubic_mse_total
        / total_batches
    )

    residual_psnr = calculate_psnr(
        residual_mse
    )

    bicubic_psnr = calculate_psnr(
        bicubic_mse
    )

    residual_ssim = (
        residual_ssim_total
        / total_batches
    )

    bicubic_ssim = (
        bicubic_ssim_total
        / total_batches
    )

    # ========================================================
    # IMPROVEMENTS
    # ========================================================

    mse_improvement = (
        (
            bicubic_mse
            - residual_mse
        )
        / bicubic_mse
        * 100
    )

    psnr_improvement = (
        residual_psnr
        - bicubic_psnr
    )

    ssim_improvement = (
        residual_ssim
        - bicubic_ssim
    )

    # ========================================================
    # RESULTS DATA
    # ========================================================

    results = [

        {
            "method": "Bicubic",
            "mse": float(
                bicubic_mse
            ),
            "psnr_db": float(
                bicubic_psnr
            ),
            "ssim": float(
                bicubic_ssim
            ),
        },

        {
            "method": "ResidualSR",
            "mse": float(
                residual_mse
            ),
            "psnr_db": float(
                residual_psnr
            ),
            "ssim": float(
                residual_ssim
            ),
        },

    ]

    improvements = {

        "mse_improvement_percent": float(
            mse_improvement
        ),

        "psnr_improvement_db": float(
            psnr_improvement
        ),

        "ssim_improvement": float(
            ssim_improvement
        ),

    }

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print()

    print(
        f"{'Method':<20}"
        f"{'MSE':<15}"
        f"{'PSNR':<15}"
        f"{'SSIM':<15}"
    )

    print("-" * 60)

    for result in results:

        print(
            f"{result['method']:<20}"
            f"{result['mse']:<15.8f}"
            f"{result['psnr_db']:<15.4f}"
            f"{result['ssim']:<15.4f}"
        )

    # ========================================================
    # PRINT IMPROVEMENTS
    # ========================================================

    print()

    print("=" * 60)
    print("RESIDUALSR IMPROVEMENT OVER BICUBIC")
    print("=" * 60)

    print()

    print(
        f"MSE improvement: "
        f"{mse_improvement:.2f}%"
    )

    print(
        f"PSNR improvement: "
        f"{psnr_improvement:.4f} dB"
    )

    print(
        f"SSIM improvement: "
        f"{ssim_improvement:.4f}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    print()

    print("=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    print()

    save_results(
        results,
        improvements,
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print("=" * 60)
    print("RESIDUALSR EVALUATION COMPLETE")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()