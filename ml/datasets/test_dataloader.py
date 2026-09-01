from ml.datasets.dataloader import create_dataloaders


def main():

    print("Creating DataLoaders...")
    print()

    train_loader, val_loader, test_loader = (
        create_dataloaders(
            batch_size=8,
            num_workers=0,
        )
    )

    print("=" * 50)
    print("DATALOADER INFORMATION")
    print("=" * 50)

    print()
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    print()

    # Get one training batch
    batch = next(iter(train_loader))

    lr = batch["lr"]
    hr = batch["hr"]

    print("=" * 50)
    print("TRAIN BATCH")
    print("=" * 50)

    print()
    print("LR batch shape:", lr.shape)
    print("HR batch shape:", hr.shape)

    print()
    print("LR dtype:", lr.dtype)
    print("HR dtype:", hr.dtype)

    print()
    print("LR range:")
    print("Min:", lr.min().item())
    print("Max:", lr.max().item())

    print()
    print("HR range:")
    print("Min:", hr.min().item())
    print("Max:", hr.max().item())

    print()
    print("SUCCESS: DataLoader is working!")


if __name__ == "__main__":
    main()