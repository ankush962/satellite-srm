from dataset import Sen2VenusDataset

import torch


ROOT_DIR = "data/raw/sen2venus/ALSACE"


def check_tensor(name, tensor):
    """Check a tensor for invalid values and basic statistics."""

    print(f"\n{name} Statistics")
    print("-" * 40)

    print("Shape:", tensor.shape)
    print("Min:", tensor.min().item())
    print("Max:", tensor.max().item())
    print("Mean:", tensor.mean().item())
    print("Std:", tensor.std().item())

    print("NaN values:", torch.isnan(tensor).sum().item())
    print("Inf values:", torch.isinf(tensor).sum().item())

    print(
        "Negative values:",
        (tensor < 0).sum().item()
    )


def main():

    print("Loading dataset...\n")

    dataset = Sen2VenusDataset(
        root_dir=ROOT_DIR,
        normalize=True
    )

    print("Total samples:", len(dataset))

    # Dataset-wide statistics
    global_lr_min = float("inf")
    global_lr_max = float("-inf")

    global_hr_min = float("inf")
    global_hr_max = float("-inf")

    total_lr_negative = 0
    total_hr_negative = 0

    total_lr_nan = 0
    total_hr_nan = 0

    total_lr_inf = 0
    total_hr_inf = 0

    corrupted_samples = []

    lr_samples_with_negatives = []
    hr_samples_with_negatives = []

    print("\nChecking all samples...")
    print("=" * 50)

    for index in range(len(dataset)):

        try:

            sample = dataset[index]

            lr = sample["lr"]
            hr = sample["hr"]

            # -------------------------
            # Check NaN values
            # -------------------------

            lr_nan = torch.isnan(lr).sum().item()
            hr_nan = torch.isnan(hr).sum().item()

            total_lr_nan += lr_nan
            total_hr_nan += hr_nan

            # -------------------------
            # Check Inf values
            # -------------------------

            lr_inf = torch.isinf(lr).sum().item()
            hr_inf = torch.isinf(hr).sum().item()

            total_lr_inf += lr_inf
            total_hr_inf += hr_inf

            # -------------------------
            # Check negative values
            # -------------------------

            lr_negative = (lr < 0).sum().item()
            hr_negative = (hr < 0).sum().item()

            total_lr_negative += lr_negative
            total_hr_negative += hr_negative

            if lr_negative > 0:
                lr_samples_with_negatives.append(
                    {
                        "index": index,
                        "negative_pixels": lr_negative,
                        "path": sample["lr_path"],
                    }
                )

            if hr_negative > 0:
                hr_samples_with_negatives.append(
                    {
                        "index": index,
                        "negative_pixels": hr_negative,
                        "path": sample["hr_path"],
                    }
                )

            # -------------------------
            # Update global min/max
            # -------------------------

            global_lr_min = min(
                global_lr_min,
                lr.min().item()
            )

            global_lr_max = max(
                global_lr_max,
                lr.max().item()
            )

            global_hr_min = min(
                global_hr_min,
                hr.min().item()
            )

            global_hr_max = max(
                global_hr_max,
                hr.max().item()
            )

            # Progress indicator
            if (index + 1) % 100 == 0:

                print(
                    f"Checked {index + 1}/{len(dataset)} samples"
                )

        except Exception as error:

            corrupted_samples.append(
                {
                    "index": index,
                    "error": str(error),
                }
            )

            print(
                f"\nERROR at sample {index}: {error}"
            )

    # =========================================
    # FINAL RESULTS
    # =========================================

    print("\n")
    print("=" * 60)
    print("DATASET VALIDATION RESULTS")
    print("=" * 60)

    print("\nDataset size:")
    print(len(dataset))

    print("\nLR (Sentinel-2)")
    print("-" * 40)
    print("Global minimum:", global_lr_min)
    print("Global maximum:", global_lr_max)
    print("Total negative pixels:", total_lr_negative)
    print("Samples with negatives:", len(lr_samples_with_negatives))
    print("Total NaN values:", total_lr_nan)
    print("Total Inf values:", total_lr_inf)

    print("\nHR (VENµS)")
    print("-" * 40)
    print("Global minimum:", global_hr_min)
    print("Global maximum:", global_hr_max)
    print("Total negative pixels:", total_hr_negative)
    print("Samples with negatives:", len(hr_samples_with_negatives))
    print("Total NaN values:", total_hr_nan)
    print("Total Inf values:", total_hr_inf)

    print("\nCorrupted samples:")
    print(len(corrupted_samples))

    # =========================================
    # SHOW NEGATIVE SAMPLE EXAMPLES
    # =========================================

    print("\n")
    print("=" * 60)
    print("LR SAMPLES WITH NEGATIVE VALUES")
    print("=" * 60)

    for sample in lr_samples_with_negatives[:10]:

        print(
            f"\nIndex: {sample['index']}"
        )

        print(
            f"Negative pixels: "
            f"{sample['negative_pixels']}"
        )

        print(
            f"Path: {sample['path']}"
        )

    print("\n")
    print("=" * 60)
    print("HR SAMPLES WITH NEGATIVE VALUES")
    print("=" * 60)

    for sample in hr_samples_with_negatives[:10]:

        print(
            f"\nIndex: {sample['index']}"
        )

        print(
            f"Negative pixels: "
            f"{sample['negative_pixels']}"
        )

        print(
            f"Path: {sample['path']}"
        )

    # =========================================
    # CORRUPTED SAMPLE DETAILS
    # =========================================

    if corrupted_samples:

        print("\n")
        print("=" * 60)
        print("CORRUPTED SAMPLE DETAILS")
        print("=" * 60)

        for sample in corrupted_samples:

            print(
                f"\nIndex: {sample['index']}"
            )

            print(
                f"Error: {sample['error']}"
            )

    print("\nValidation complete.")


if __name__ == "__main__":
    main()