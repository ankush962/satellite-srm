from torch.utils.data import DataLoader, Subset

from ml.datasets.dataset import Sen2VenusDataset


def load_indices(file_path):
    """Load dataset indices from a split file."""

    with open(file_path, "r") as f:
        indices = [
            int(line.strip())
            for line in f
            if line.strip()
        ]

    return indices


def create_dataloaders(
    data_dir="data/raw/sen2venus/ALSACE",
    splits_dir="data/splits",
    batch_size=8,
    num_workers=0,
):
    """Create train, validation, and test DataLoaders."""

    dataset = Sen2VenusDataset(data_dir)

    train_indices = load_indices(
        f"{splits_dir}/train.txt"
    )

    val_indices = load_indices(
        f"{splits_dir}/val.txt"
    )

    test_indices = load_indices(
        f"{splits_dir}/test.txt"
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader