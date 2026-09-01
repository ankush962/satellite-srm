import sys
from pathlib import Path

import torch

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from ml.models.baseline import BaselineSR


def main():

    model = BaselineSR()

    print("=" * 50)
    print("BASELINE MODEL TEST")
    print("=" * 50)

    print()

    # Fake Sentinel-2 batch
    x = torch.rand(
        8,
        4,
        128,
        128,
    )

    print("Input shape:")
    print(x.shape)

    print()

    # Forward pass
    with torch.no_grad():
        output = model(x)

    print("Output shape:")
    print(output.shape)

    print()

    print("Expected output:")
    print(torch.Size([8, 4, 256, 256]))

    print()

    # Verify
    assert output.shape == (
        8,
        4,
        256,
        256,
    )

    print("SUCCESS: Model output shape is correct!")

    print()

    # Count parameters
    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Trainable parameters:")
    print(parameters)


if __name__ == "__main__":
    main()