from dataset import Sen2VenusDataset


dataset = Sen2VenusDataset(
    root_dir="data/raw/sen2venus/ALSACE"
)

print("Dataset size:", len(dataset))
print()

lr_global_min = float("inf")
lr_global_max = float("-inf")

hr_global_min = float("inf")
hr_global_max = float("-inf")


print("Checking dataset normalization...")
print("=" * 50)


for i in range(len(dataset)):

    sample = dataset[i]

    lr = sample["lr"]
    hr = sample["hr"]

    lr_global_min = min(lr_global_min, lr.min().item())
    lr_global_max = max(lr_global_max, lr.max().item())

    hr_global_min = min(hr_global_min, hr.min().item())
    hr_global_max = max(hr_global_max, hr.max().item())

    if (i + 1) % 200 == 0:
        print(f"Checked {i + 1}/{len(dataset)} samples")


print()
print("=" * 50)
print("NORMALIZATION RESULTS")
print("=" * 50)

print()
print("LR Sentinel-2")
print("Minimum:", lr_global_min)
print("Maximum:", lr_global_max)

print()
print("HR VENµS")
print("Minimum:", hr_global_min)
print("Maximum:", hr_global_max)

print()
print("=" * 50)

if (
    lr_global_min >= 0
    and lr_global_max <= 1
    and hr_global_min >= 0
    and hr_global_max <= 1
):
    print("SUCCESS: All values are in range [0, 1]")
else:
    print("WARNING: Values outside [0, 1] detected")