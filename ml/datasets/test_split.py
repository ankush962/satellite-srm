from pathlib import Path


SPLIT_DIR = Path("data/splits")


def load_indices(filename):
    path = SPLIT_DIR / filename

    with open(path, "r") as f:
        return [
            int(line.strip())
            for line in f
            if line.strip()
        ]


def main():

    train_indices = load_indices("train.txt")
    val_indices = load_indices("val.txt")
    test_indices = load_indices("test.txt")

    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)

    print("=" * 50)
    print("SPLIT VERIFICATION")
    print("=" * 50)

    print()
    print(f"Train samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Test samples: {len(test_indices)}")

    print()
    print("Checking overlaps...")
    print()

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    print(f"Train ∩ Validation: {len(train_val_overlap)}")
    print(f"Train ∩ Test: {len(train_test_overlap)}")
    print(f"Validation ∩ Test: {len(val_test_overlap)}")

    all_indices = train_set | val_set | test_set
    total_split_samples = (
        len(train_indices)
        + len(val_indices)
        + len(test_indices)
    )

    print()
    print(f"Unique indices: {len(all_indices)}")
    print(f"Total split entries: {total_split_samples}")

    print()

    no_overlap = (
        len(train_val_overlap) == 0
        and len(train_test_overlap) == 0
        and len(val_test_overlap) == 0
    )

    no_duplicates = (
        len(train_set) == len(train_indices)
        and len(val_set) == len(val_indices)
        and len(test_set) == len(test_indices)
    )

    complete_split = (
        len(all_indices) == total_split_samples
    )

    if no_overlap and no_duplicates and complete_split:
        print("SUCCESS: Dataset split is valid!")
    else:
        print("ERROR: Problem detected in dataset split!")

        if not no_overlap:
            print("\nProblem: Overlapping indices detected.")

        if not no_duplicates:
            print("\nProblem: Duplicate indices detected.")

        if not complete_split:
            print("\nProblem: Missing or duplicate indices detected.")


if __name__ == "__main__":
    main()
