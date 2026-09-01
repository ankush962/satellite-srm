import csv
import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASELINE_RESULTS_PATH = Path(
    "results/metrics/comparison_results.json"
)

RESIDUAL_RESULTS_PATH = Path(
    "results/metrics/residual_comparison_results.json"
)

OUTPUT_DIR = Path(
    "results/metrics"
)


# ============================================================
# LOAD JSON RESULTS
# ============================================================

def load_json(path):
    """
    Load evaluation results from a JSON file.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Results file not found: {path}"
        )

    with open(
        path,
        "r",
    ) as file:

        return json.load(file)


# ============================================================
# EXTRACT RESULTS
# ============================================================

def get_result_by_method(
    results,
    method_name,
):
    """
    Find a result dictionary by method name.
    """

    for result in results:

        if result["method"] == method_name:

            return result

    raise ValueError(
        f"Method '{method_name}' not found."
    )


# ============================================================
# SAVE FINAL JSON
# ============================================================

def save_json(
    results,
    improvements,
):
    """
    Save final comparison results as JSON.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "final_model_comparison.json"
    )

    data = {
        "results": results,
        "improvements": improvements,
    }

    with open(
        output_path,
        "w",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
        )

    print(
        f"Saved JSON: {output_path}"
    )


# ============================================================
# SAVE FINAL CSV
# ============================================================

def save_csv(results):
    """
    Save final comparison results as CSV.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "final_model_comparison.csv"
    )

    with open(
        output_path,
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
        f"Saved CSV: {output_path}"
    )


# ============================================================
# CALCULATE IMPROVEMENTS
# ============================================================

def calculate_improvement(
    bicubic,
    model,
):
    """
    Calculate model improvement over Bicubic.
    """

    mse_improvement = (
        (
            bicubic["mse"]
            - model["mse"]
        )
        / bicubic["mse"]
        * 100
    )

    psnr_improvement = (
        model["psnr_db"]
        - bicubic["psnr_db"]
    )

    ssim_improvement = (
        model["ssim"]
        - bicubic["ssim"]
    )

    return {
        "mse_improvement_percent":
            mse_improvement,

        "psnr_improvement_db":
            psnr_improvement,

        "ssim_improvement":
            ssim_improvement,
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    results,
    improvements,
):

    print()

    print("=" * 70)

    print(
        "FINAL THREE-MODEL COMPARISON"
    )

    print("=" * 70)

    print()

    print(
        f"{'Method':<20}"
        f"{'MSE':<18}"
        f"{'PSNR (dB)':<18}"
        f"{'SSIM':<15}"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['method']:<20}"
            f"{result['mse']:<18.8f}"
            f"{result['psnr_db']:<18.4f}"
            f"{result['ssim']:<15.4f}"
        )

    print()

    print("=" * 70)

    print(
        "IMPROVEMENT OVER BICUBIC"
    )

    print("=" * 70)

    print()

    for model_name, values in improvements.items():

        print(
            f"{model_name}"
        )

        print(
            f"  MSE improvement: "
            f"{values['mse_improvement_percent']:.2f}%"
        )

        print(
            f"  PSNR improvement: "
            f"{values['psnr_improvement_db']:.4f} dB"
        )

        print(
            f"  SSIM improvement: "
            f"{values['ssim_improvement']:.4f}"
        )

        print()


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)

    print(
        "LOADING MODEL EVALUATION RESULTS"
    )

    print("=" * 70)

    print()

    # --------------------------------------------------------
    # LOAD BASELINE RESULTS
    # --------------------------------------------------------

    print(
        "Loading Baseline CNN results..."
    )

    baseline_data = load_json(
        BASELINE_RESULTS_PATH
    )

    baseline_results = (
        baseline_data["results"]
    )

    # --------------------------------------------------------
    # LOAD RESIDUAL RESULTS
    # --------------------------------------------------------

    print(
        "Loading ResidualSR results..."
    )

    residual_data = load_json(
        RESIDUAL_RESULTS_PATH
    )

    residual_results = (
        residual_data["results"]
    )

    # --------------------------------------------------------
    # EXTRACT INDIVIDUAL MODELS
    # --------------------------------------------------------

    bicubic = get_result_by_method(
        baseline_results,
        "Bicubic",
    )

    baseline = get_result_by_method(
        baseline_results,
        "Baseline CNN",
    )

    residual = get_result_by_method(
        residual_results,
        "ResidualSR",
    )

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    final_results = [

        {
            "method":
                "Bicubic",

            "mse":
                float(bicubic["mse"]),

            "psnr_db":
                float(bicubic["psnr_db"]),

            "ssim":
                float(bicubic["ssim"]),
        },

        {
            "method":
                "Baseline CNN",

            "mse":
                float(baseline["mse"]),

            "psnr_db":
                float(baseline["psnr_db"]),

            "ssim":
                float(baseline["ssim"]),
        },

        {
            "method":
                "ResidualSR",

            "mse":
                float(residual["mse"]),

            "psnr_db":
                float(residual["psnr_db"]),

            "ssim":
                float(residual["ssim"]),
        },

    ]

    # --------------------------------------------------------
    # CALCULATE IMPROVEMENTS
    # --------------------------------------------------------

    improvements = {

        "Baseline CNN":

            calculate_improvement(
                bicubic,
                baseline,
            ),

        "ResidualSR":

            calculate_improvement(
                bicubic,
                residual,
            ),

    }

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print_results(
        final_results,
        improvements,
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    print("=" * 70)

    print(
        "SAVING FINAL COMPARISON RESULTS"
    )

    print("=" * 70)

    print()

    save_json(
        final_results,
        improvements,
    )

    save_csv(
        final_results
    )

    print()

    print("=" * 70)

    print(
        "FINAL COMPARISON COMPLETE"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()