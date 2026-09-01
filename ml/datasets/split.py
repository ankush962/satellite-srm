from pathlib import Path
import random

from ml.datasets.dataset import Sen2VenusDataset


ROOT_DIR = Path("data/raw/sen2venus/ALSACE")
OUTPUT_DIR = Path("data/splits")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42


def get_date_from_path(path_string):
    """
    Extract acquisition date from a dataset file path.

    Example:
    data/raw/sen2venus/ALSACE/2019-09-15/.../file_10m.tif

    Returns:
    2019-09-15
    """

    path = Path(path_string)

    for part in path.parts:
        if (
            len(part) == 10
            and part[4] == "-"
            and part[7] == "-"
        ):
            return part

    raise ValueError(
        f"Could not find date in path: {path}"
    )


def main():

    print()
    print("=" * 60)
    print("CREATING GEOGRAPHIC / DATE-BASED DATASET SPLIT")
    print("=" * 60)

    print()
    print("Loading dataset...")

    dataset = Sen2VenusDataset(ROOT_DIR)

    total_samples = len(dataset)

    print(f"Total samples: {total_samples}")

    # --------------------------------------------------------
    # GROUP DATASET INDICES BY ACQUISITION DATE
    # --------------------------------------------------------

    date_to_indices = {}

    for index, (lr_path, hr_path) in enumerate(dataset.pairs):

        date = get_date_from_path(lr_path)

        if date not in date_to_indices:
            date_to_indices[date] = []

        date_to_indices[date].append(index)

    dates = sorted(date_to_indices.keys())

    print()
    print(f"Total acquisition dates: {len(dates)}")

    print()
    print("Samples per date:")

    for date in dates:
        print(
            f"  {date}: "
            f"{len(date_to_indices[date])} samples"
        )

    # --------------------------------------------------------
    # SHUFFLE DATES, NOT INDIVIDUAL PATCHES
    # --------------------------------------------------------

    shuffled_dates = dates.copy()

    random.seed(SEED)
    random.shuffle(shuffled_dates)

    total_dates = len(shuffled_dates)

    train_date_count = int(total_dates * TRAIN_RATIO)
    val_date_count = int(total_dates * VAL_RATIO)

    # Remaining dates go to testing
    test_date_count = (
        total_dates
        - train_date_count
        - val_date_count
    )

    train_dates = shuffled_dates[:train_date_count]

    val_dates = shuffled_dates[
        train_date_count:
        train_date_count + val_date_count
    ]

    test_dates = shuffled_dates[
        train_date_count + val_date_count:
    ]

    # --------------------------------------------------------
    # CREATE SAMPLE INDICES
    # --------------------------------------------------------

    train_indices = []

    for date in train_dates:
        train_indices.extend(
            date_to_indices[date]
        )

    val_indices = []

    for date in val_dates:
        val_indices.extend(
            date_to_indices[date]
        )

    test_indices = []

    for date in test_dates:
        test_indices.extend(
            date_to_indices[date]
        )

    # Sort indices for reproducibility
    train_indices.sort()
    val_indices.sort()
    test_indices.sort()

    # --------------------------------------------------------
    # SAFETY CHECKS
    # --------------------------------------------------------

    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)

    assert len(
        train_set & val_set
    ) == 0, "Train/Validation overlap detected!"

    assert len(
        train_set & test_set
    ) == 0, "Train/Test overlap detected!"

    assert len(
        val_set & test_set
    ) == 0, "Validation/Test overlap detected!"

    assert (
        len(train_indices)
        + len(val_indices)
        + len(test_indices)
        == total_samples
    ), "Some samples are missing!"

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SAVE SPLITS
    # --------------------------------------------------------

    splits = {
        "train.txt": train_indices,
        "val.txt": val_indices,
        "test.txt": test_indices,
    }

    for filename, indices in splits.items():

        output_path = OUTPUT_DIR / filename

        with open(output_path, "w") as file:

            for index in indices:
                file.write(f"{index}\n")

        print()
        print(
            f"Saved {filename}: "
            f"{len(indices)} samples"
        )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DATE-BASED DATASET SPLIT COMPLETE")
    print("=" * 60)

    print()
    print("TRAIN DATES:")

    for date in sorted(train_dates):
        print(
            f"  {date} "
            f"({len(date_to_indices[date])} samples)"
        )

    print()
    print("VALIDATION DATES:")

    for date in sorted(val_dates):
        print(
            f"  {date} "
            f"({len(date_to_indices[date])} samples)"
        )

    print()
    print("TEST DATES:")

    for date in sorted(test_dates):
        print(
            f"  {date} "
            f"({len(date_to_indices[date])} samples)"
        )

    print()
    print("-" * 60)

    print(
        f"Train samples: {len(train_indices)} "
        f"({len(train_dates)} dates)"
    )

    print(
        f"Validation samples: {len(val_indices)} "
        f"({len(val_dates)} dates)"
    )

    print(
        f"Test samples: {len(test_indices)} "
        f"({len(test_dates)} dates)"
    )

    print(
        f"Total samples: {total_samples}"
    )

    print()
    print(f"Random seed: {SEED}")

    print()
    print(
        "IMPORTANT: No acquisition date appears "
        "in more than one split."
    )

    print()
    print(
        f"Split files saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()