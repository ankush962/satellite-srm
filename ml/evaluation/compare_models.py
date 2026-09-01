import json
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from ml.datasets.dataloader import create_dataloaders
from ml.models.baseline import BaselineSR
from ml.models.residual_sr import ResidualSR


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

BASELINE_CHECKPOINT = "checkpoints/best_baseline_model.pth"
RESIDUAL_CHECKPOINT = "checkpoints/best_residual_geo_model.pth"

RESULTS_DIR = Path("results/metrics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(pred, target):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    mse = float(np.mean((pred_np - target_np) ** 2))
    psnr = float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf")

    ssim_values = []
    for i in range(pred_np.shape[0]):
        pred_img = np.transpose(pred_np[i], (1, 2, 0))
        target_img = np.transpose(target_np[i], (1, 2, 0))

        score = ssim(
            target_img,
            pred_img,
            channel_axis=2,
            data_range=1.0,
        )
        ssim_values.append(float(score))

    return mse, psnr, float(np.mean(ssim_values))


def bicubic_upscale(x):
    return torch.nn.functional.interpolate(
        x,
        scale_factor=2,
        mode="bicubic",
        align_corners=False,
    ).clamp(0.0, 1.0)


def evaluate_model(model, test_loader):
    model.eval()

    mse_values = []
    psnr_values = []
    ssim_values = []

    with torch.no_grad():
        for batch in test_loader:
            lr = batch["lr"].to(DEVICE)
            hr = batch["hr"].to(DEVICE)

            pred = model(lr).clamp(0.0, 1.0)

            mse, psnr, ssim_score = calculate_metrics(pred, hr)

            mse_values.append(mse)
            psnr_values.append(psnr)
            ssim_values.append(ssim_score)

    return {
        "mse": float(np.mean(mse_values)),
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
    }


def evaluate_bicubic(test_loader):
    mse_values = []
    psnr_values = []
    ssim_values = []

    with torch.no_grad():
        for batch in test_loader:
            lr = batch["lr"].to(DEVICE)
            hr = batch["hr"].to(DEVICE)

            pred = bicubic_upscale(lr)

            mse, psnr, ssim_score = calculate_metrics(pred, hr)

            mse_values.append(mse)
            psnr_values.append(psnr)
            ssim_values.append(ssim_score)

    return {
        "mse": float(np.mean(mse_values)),
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
    }


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


def main():
    print("=" * 65)
    print("DATE-BASED FINAL MODEL EVALUATION")
    print("=" * 65)
    print(f"Device: {DEVICE}")

    _, _, test_loader = create_dataloaders(
        batch_size=8,
        num_workers=0,
    )

    print(f"Test batches: {len(test_loader)}")
    print()

    baseline_model = BaselineSR(
        in_channels=4,
        out_channels=4,
        features=64,
    )
    baseline_model = load_checkpoint(
        baseline_model,
        BASELINE_CHECKPOINT,
    ).to(DEVICE)

    residual_model = ResidualSR(
        in_channels=4,
        out_channels=4,
        features=64,
        num_blocks=8,
    )
    residual_model = load_checkpoint(
        residual_model,
        RESIDUAL_CHECKPOINT,
    ).to(DEVICE)

    print("Evaluating Bicubic...")
    bicubic_results = evaluate_bicubic(test_loader)

    print("Evaluating Baseline CNN...")
    baseline_results = evaluate_model(
        baseline_model,
        test_loader,
    )

    print("Evaluating ResidualSR...")
    residual_results = evaluate_model(
        residual_model,
        test_loader,
    )

    results = {
        "dataset_split": "date_based",
        "test_samples": 546,
        "models": {
            "Bicubic": bicubic_results,
            "Baseline_CNN": baseline_results,
            "ResidualSR": residual_results,
        },
    }

    results["improvements"] = {
        "Baseline_vs_Bicubic": {
            "mse_percent": float(
                (1 - baseline_results["mse"] / bicubic_results["mse"]) * 100
            ),
            "psnr_db": float(
                baseline_results["psnr"] - bicubic_results["psnr"]
            ),
            "ssim": float(
                baseline_results["ssim"] - bicubic_results["ssim"]
            ),
        },
        "ResidualSR_vs_Bicubic": {
            "mse_percent": float(
                (1 - residual_results["mse"] / bicubic_results["mse"]) * 100
            ),
            "psnr_db": float(
                residual_results["psnr"] - bicubic_results["psnr"]
            ),
            "ssim": float(
                residual_results["ssim"] - bicubic_results["ssim"]
            ),
        },
        "ResidualSR_vs_Baseline": {
            "mse_percent": float(
                (1 - residual_results["mse"] / baseline_results["mse"]) * 100
            ),
            "psnr_db": float(
                residual_results["psnr"] - baseline_results["psnr"]
            ),
            "ssim": float(
                residual_results["ssim"] - baseline_results["ssim"]
            ),
        },
    }

    output_path = RESULTS_DIR / "date_based_final_comparison.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 65)
    print("RESULTS")
    print("=" * 65)

    for name, metrics in results["models"].items():
        print(f"\n{name}")
        print(f"  MSE :  {metrics['mse']:.8f}")
        print(f"  PSNR:  {metrics['psnr']:.4f} dB")
        print(f"  SSIM:  {metrics['ssim']:.4f}")

    print("\nResidualSR vs Bicubic")
    print(
        f"  MSE improvement : "
        f"{results['improvements']['ResidualSR_vs_Bicubic']['mse_percent']:.2f}%"
    )
    print(
        f"  PSNR improvement: "
        f"{results['improvements']['ResidualSR_vs_Bicubic']['psnr_db']:.4f} dB"
    )
    print(
        f"  SSIM improvement: "
        f"{results['improvements']['ResidualSR_vs_Bicubic']['ssim']:.4f}"
    )

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
