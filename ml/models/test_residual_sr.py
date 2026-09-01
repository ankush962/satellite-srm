import torch

from ml.models.residual_sr import ResidualSR


def main():

    print("=" * 60)
    print("RESIDUAL SUPER-RESOLUTION MODEL TEST")
    print("=" * 60)

    model = ResidualSR()

    print()
    print("Model created successfully!")

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print(
        f"Trainable parameters: "
        f"{total_parameters:,}"
    )

    print()

    # --------------------------------------------------------
    # TEST INPUT
    # --------------------------------------------------------

    x = torch.rand(
        2,
        4,
        128,
        128,
    )

    print(
        f"Input shape:  {x.shape}"
    )

    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(x)

    print(
        f"Output shape: {output.shape}"
    )

    # --------------------------------------------------------
    # VERIFY
    # --------------------------------------------------------

    expected_shape = (
        2,
        4,
        256,
        256,
    )

    print()

    if output.shape == expected_shape:

        print(
            "SUCCESS: Model output shape is correct!"
        )

    else:

        print(
            "ERROR: Unexpected output shape!"
        )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()