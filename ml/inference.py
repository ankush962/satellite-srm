from pathlib import Path

import numpy as np
import rasterio
import torch

from ml.models.residual_sr import ResidualSR


CHECKPOINT = "checkpoints/best_residual_geo_model.pth"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model():
    model = ResidualSR(
        in_channels=4,
        out_channels=4,
        features=64,
        num_blocks=8,
        scale_factor=2,
    )

    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


def super_resolve(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    model = load_model()

    with rasterio.open(input_path) as src:
        image = src.read().astype(np.float32)

        if image.shape[0] != 4:
            raise ValueError(
                f"Expected 4 bands (B2, B3, B4, B8), got {image.shape[0]}"
            )

        profile = src.profile.copy()

        image = image / 10000.0
        image = np.clip(image, 0.0, 1.0)

        tensor = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prediction = model(tensor).clamp(0.0, 1.0)

        prediction = prediction.squeeze(0).cpu().numpy()

        prediction = np.clip(prediction * 10000.0, 0, 10000).astype(np.uint16)

        height, width = prediction.shape[1], prediction.shape[2]

        transform = src.transform

        new_transform = rasterio.Affine(
            transform.a / 2,
            transform.b,
            transform.c,
            transform.d,
            transform.e / 2,
            transform.f,
        )

        profile.update(
            height=height,
            width=width,
            transform=new_transform,
            dtype="uint16",
            count=4,
            compress="deflate",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(prediction)

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Device: {DEVICE}")
    print(f"Output shape: {prediction.shape}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Deep learning satellite image super-resolution"
    )

    parser.add_argument("input", help="Input 10m 4-band GeoTIFF")
    parser.add_argument("output", help="Output 5m super-resolved GeoTIFF")

    args = parser.parse_args()

    super_resolve(args.input, args.output)
