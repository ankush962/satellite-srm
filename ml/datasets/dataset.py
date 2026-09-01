from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class Sen2VenusDataset(Dataset):
    """
    Dataset for paired Sentinel-2 (10m) and VENµS (5m) patches.

    LR:
        4 bands, 128 x 128

    HR:
        4 bands, 256 x 256
    """

    def __init__(self, root_dir, normalize=True):
        self.root_dir = Path(root_dir)
        self.normalize = normalize

        self.pairs = self._find_pairs()

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No LR/HR pairs found in: {self.root_dir}"
            )

    def _find_pairs(self):
        """Find all matching 10m and 05m TIFF files recursively."""

        lr_files = sorted(self.root_dir.rglob("*_10m.tif"))
        pairs = []

        for lr_path in lr_files:
            hr_name = lr_path.name.replace("_10m.tif", "_05m.tif")

            hr_path = (
                lr_path.parent.parent
                / "05m"
                / hr_name
            )

            if hr_path.exists():
                pairs.append((lr_path, hr_path))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def _read_tiff(self, path):
        with rasterio.open(path) as src:
            image = src.read()

        image = image.astype(np.float32)

        if self.normalize:
            # Convert reflectance scale
            image = image / 10000.0

            # Remove negative values and extreme values
            image = np.clip(image, 0.0, 1.0)

        return image

    def __getitem__(self, index):
        lr_path, hr_path = self.pairs[index]

        lr_image = self._read_tiff(lr_path)
        hr_image = self._read_tiff(hr_path)

        lr_tensor = torch.from_numpy(lr_image)
        hr_tensor = torch.from_numpy(hr_image)

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "lr_path": str(lr_path),
            "hr_path": str(hr_path),
        }
