import json
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

RESULTS_PATH = Path(
    "results/metrics/final_model_comparison.json"
)

OUTPUT_DIR = Path(
    "results/plots"
)


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():
    """
    Load final model comparison results.
    """

    if not RESULTS_PATH.exists():

        raise FileNotFoundError(
            f"Results file not found: {RESULTS_PATH}"
        )

    with open(
        RESULTS_PATH,
        "r",
    ) as file:

        data = json.load(file)

    return data["results"]


# ============================================================
# CREATE MSE PLOT
# ============================================================

def plot_mse(
    methods,
    mse_values,
):
    """
    Create MSE comparison chart.
    """

    plt.figure(
        figsize=(8, 6)
    )

    bars = plt.bar(
        methods,
        mse_values,
    )

    plt.title(
        "Model Comparison: Mean Squared Error"
    )

    plt.xlabel(
        "Method"
    )

    plt.ylabel(
        "MSE"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    for bar, value in zip(
        bars,
        mse_values,
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value,
            f"{value:.6f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "mse_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved plot: {output_path}"
    )


# ============================================================
# CREATE PSNR PLOT
# ============================================================

def plot_psnr(
    methods,
    psnr_values,
):
    """
    Create PSNR comparison chart.
    """

    plt.figure(
        figsize=(8, 6)
    )

    bars = plt.bar(
        methods,
        psnr_values,
    )

    plt.title(
        "Model Comparison: PSNR"
    )

    plt.xlabel(
        "Method"
    )

    plt.ylabel(
        "PSNR (dB)"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    for bar, value in zip(
        bars,
        psnr_values,
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "psnr_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved plot: {output_path}"
    )


# ============================================================
# CREATE SSIM PLOT
# ============================================================

def plot_ssim(
    methods,
    ssim_values,
):
    """
    Create SSIM comparison chart.
    """

    plt.figure(
        figsize=(8, 6)
    )

    bars = plt.bar(
        methods,
        ssim_values,
    )

    plt.title(
        "Model Comparison: Structural Similarity"
    )

    plt.xlabel(
        "Method"
    )

    plt.ylabel(
        "SSIM"
    )

    plt.ylim(
        min(ssim_values) - 0.02,
        1.0,
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    for bar, value in zip(
        bars,
        ssim_values,
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "ssim_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved plot: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 60)

    print(
        "GENERATING MODEL COMPARISON PLOTS"
    )

    print("=" * 60)

    print()

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD RESULTS
    # --------------------------------------------------------

    print(
        "Loading comparison results..."
    )

    results = load_results()

    # --------------------------------------------------------
    # EXTRACT VALUES
    # --------------------------------------------------------

    methods = [
        result["method"]
        for result in results
    ]

    mse_values = [
        result["mse"]
        for result in results
    ]

    psnr_values = [
        result["psnr_db"]
        for result in results
    ]

    ssim_values = [
        result["ssim"]
        for result in results
    ]

    # --------------------------------------------------------
    # PRINT DATA
    # --------------------------------------------------------

    print()

    print(
        "Models found:"
    )

    for method in methods:

        print(
            f"  - {method}"
        )

    print()

    # --------------------------------------------------------
    # GENERATE PLOTS
    # --------------------------------------------------------

    print(
        "Generating MSE comparison..."
    )

    plot_mse(
        methods,
        mse_values,
    )

    print()

    print(
        "Generating PSNR comparison..."
    )

    plot_psnr(
        methods,
        psnr_values,
    )

    print()

    print(
        "Generating SSIM comparison..."
    )

    plot_ssim(
        methods,
        ssim_values,
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print(
        "ALL COMPARISON PLOTS GENERATED SUCCESSFULLY"
    )

    print("=" * 60)

    print()

    print(
        f"Plots saved in: {OUTPUT_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()