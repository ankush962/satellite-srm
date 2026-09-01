from dataset import Sen2VenusDataset

dataset = Sen2VenusDataset(
root_dir="data/raw/sen2venus/ALSACE"
)

print("Dataset size:", len(dataset))

sample = dataset[0]

print("\nLR shape:", sample["lr"].shape)
print("HR shape:", sample["hr"].shape)

print("\nLR dtype:", sample["lr"].dtype)
print("HR dtype:", sample["hr"].dtype)

print("\nLR min:", sample["lr"].min().item())
print("LR max:", sample["lr"].max().item())

print("\nHR min:", sample["hr"].min().item())
print("HR max:", sample["hr"].max().item())

print("\nLR file:")
print(sample["lr_path"])

print("\nHR file:")
print(sample["hr_path"])

# Verify a few samples across the dataset

print("\n--- Pair verification ---")

indices = [
0,
len(dataset) // 2,
len(dataset) - 1,
]

for idx in indices:
    sample = dataset[idx]


print(f"\nIndex: {idx}")
print("LR:", sample["lr"].shape)
print("HR:", sample["hr"].shape)
print("LR path:", sample["lr_path"])
print("HR path:", sample["hr_path"])
