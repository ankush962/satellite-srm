from pathlib import Path

from ml.datasets.dataloader import create_dataloaders
from ml.datasets.dataset import Sen2VenusDataset


DATA_DIR = "data/raw/sen2venus/ALSACE"
SPLITS_DIR = "data/splits"


def get_date_from_path(path_string):
    """Extract acquisition date from a file path."""

    path = Path(path_string)

    for part in path.parts:
        if (
            len(part) == 10
            and part[4] == "-"
            and part[7] == "-"
        ):
            return part

    raise ValueError(
        f"Could not extract date from: {path}"
    )


def get_dates_from_loader(loader):
    """Get all unique acquisition dates from a DataLoader."""

    dates = set()

    for batch in loader:
        for path in batch["lr_path"]:
            dates.add(get_date_from_path(path))

    return dates


def main():

    print()
    print("=" * 60)
    print("VERIFYING DATE-BASED DATASET SPLIT")
    print("=" * 60)

    print()
    print("Creating dataloaders...")

    train_loader, val_loader, test_loader = (
        create_dataloaders(
            data_dir=DATA_DIR,
            splits_dir=SPLITS_DIR,
            batch_size=8,
            num_workers=0,
        )
    )

    print()
    print("Dataset sizes:")

    print(
        f"Train samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    print()
    print("Checking acquisition dates...")

    train_dates = get_dates_from_loader(train_loader)
    val_dates = get_dates_from_loader(val_loader)
    test_dates = get_dates_from_loader(test_loader)

    print()
    print("TRAIN DATES:")

    for date in sorted(train_dates):
        print(f"  {date}")

    print()
    print("VALIDATION DATES:")

    for date in sorted(val_dates):
        print(f"  {date}")

    print()
    print("TEST DATES:")

    for date in sorted(test_dates):
        print(f"  {date}")

    # Check for date overlap
    train_val_overlap = train_dates & val_dates
    train_test_overlap = train_dates & test_dates
    val_test_overlap = val_dates & test_dates

    print()
    print("=" * 60)
    print("OVERLAP CHECK")
    print("=" * 60)

    print()

    print(
        f"Train / Validation overlap: "
        f"{len(train_val_overlap)}"
    )

    print(
        f"Train / Test overlap: "
        f"{len(train_test_overlap)}"
    )

    print(
        f"Validation / Test overlap: "
        f"{len(val_test_overlap)}"
    )

    assert len(train_val_overlap) == 0, (
        "ERROR: Train and Validation dates overlap!"
    )

    assert len(train_test_overlap) == 0, (
        "ERROR: Train and Test dates overlap!"
    )

    assert len(val_test_overlap) == 0, (
        "ERROR: Validation and Test dates overlap!"
    )

    # Verify all samples
    dataset = Sen2VenusDataset(DATA_DIR)

    total_loaded = (
        len(train_loader.dataset)
        + len(val_loader.dataset)
        + len(test_loader.dataset)
    )

    assert total_loaded == len(dataset), (
        "ERROR: Dataset sample count mismatch!"
    )

    print()
    print("=" * 60)
    print("✓ DATE-BASED SPLIT VERIFIED SUCCESSFULLY")
    print("=" * 60)

    print()
    print(
        "No acquisition date appears in more than one split."
    )

    print(
        f"Total dataset samples verified: {total_loaded}"
    )


if __name__ == "__main__":
    main()