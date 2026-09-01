import csv
import json
from pathlib import Path


# ============================================================
# RESULTS FROM MODEL vs BICUBIC EVALUATION
# ============================================================

RESULTS = [
    {
        "method": "Bicubic",
        "mse": 0.00021517,
        "psnr_db": 36.6722,
        "ssim": 0.9427,
    },
    {
        "method": "Baseline CNN",
        "mse": 0.00015350,
        "psnr_db": 38.1389,
        "ssim": 0.9593,
    },
]


# ============================================================
# IMPROVEMENTS
# ============================================================

IMPROVEMENTS = {
    "mse_improvement_percent": 28.66,
    "psnr_improvement_db": 1.4667,
    "ssim_improvement": 0.0165,
}


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = Path("results/metrics")


def save_json():
    """Save evaluation results as JSON."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "comparison_results.json"

    data = {
        "results": RESULTS,
        "improvement_over_bicubic": IMPROVEMENTS,
    }

    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Saved JSON: {output_path}")


def save_csv():
    """Save evaluation results as CSV."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "comparison_results.csv"

    with open(
        output_path,
        "w",
        newline=""
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
        writer.writerows(RESULTS)

    print(f"Saved CSV: {output_path}")


def print_results():
    """Print final results."""

    print()
    print("=" * 60)
    print("FINAL SAVED RESULTS")
    print("=" * 60)

    print()

    print(
        f"{'Method':<20}"
        f"{'MSE':<15}"
        f"{'PSNR':<15}"
        f"{'SSIM':<15}"
    )

    print("-" * 60)

    for result in RESULTS:

        print(
            f"{result['method']:<20}"
            f"{result['mse']:<15.8f}"
            f"{result['psnr_db']:<15.4f}"
            f"{result['ssim']:<15.4f}"
        )

    print()

    print("=" * 60)
    print("IMPROVEMENT OVER BICUBIC")
    print("=" * 60)

    print()

    print(
        f"MSE improvement: "
        f"{IMPROVEMENTS['mse_improvement_percent']:.2f}%"
    )

    print(
        f"PSNR improvement: "
        f"{IMPROVEMENTS['psnr_improvement_db']:.4f} dB"
    )

    print(
        f"SSIM improvement: "
        f"{IMPROVEMENTS['ssim_improvement']:.4f}"
    )


def main():

    print("=" * 60)
    print("SAVING EVALUATION RESULTS")
    print("=" * 60)

    print()

    save_json()
    save_csv()

    print_results()

    print()
    print("=" * 60)
    print("RESULTS SAVED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()