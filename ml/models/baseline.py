import torch
import torch.nn as nn


class BaselineSR(nn.Module):
    """
    Simple 2x Super-Resolution model.

    Input:
        [B, 4, 128, 128]

    Output:
        [B, 4, 256, 256]
    """

    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        features=64,
    ):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels,
                features,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                features,
                features,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                features,
                features,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(

            nn.Conv2d(
                features,
                features * 4,
                kernel_size=3,
                padding=1,
            ),

            nn.PixelShuffle(2),

            nn.ReLU(inplace=True),
        )

        self.output = nn.Conv2d(
            features,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):

        x = self.features(x)

        x = self.upsample(x)

        x = self.output(x)


        return x