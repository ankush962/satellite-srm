import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),

        )

    def forward(self, x):

        residual = self.block(x)

        return x + residual


# ============================================================
# RESIDUAL SUPER-RESOLUTION MODEL
# ============================================================

class ResidualSR(nn.Module):

    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        features=64,
        num_blocks=8,
        scale_factor=2,
    ):

        super().__init__()

        self.scale_factor = scale_factor

        # ----------------------------------------------------
        # FEATURE EXTRACTION
        # ----------------------------------------------------

        self.input_conv = nn.Conv2d(
            in_channels,
            features,
            kernel_size=3,
            padding=1,
        )

        # ----------------------------------------------------
        # RESIDUAL BLOCKS
        # ----------------------------------------------------

        blocks = []

        for _ in range(num_blocks):

            blocks.append(
                ResidualBlock(features)
            )

        self.residual_blocks = nn.Sequential(
            *blocks
        )

        # ----------------------------------------------------
        # FEATURE RECONSTRUCTION
        # ----------------------------------------------------

        self.feature_conv = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            padding=1,
        )

        # ----------------------------------------------------
        # UPSCALING
        # ----------------------------------------------------

        self.upscale = nn.Sequential(

            nn.Conv2d(
                features,
                features * (scale_factor ** 2),
                kernel_size=3,
                padding=1,
            ),

            nn.PixelShuffle(
                scale_factor
            ),

            nn.ReLU(inplace=True),

        )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        self.output_conv = nn.Conv2d(
            features,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # BICUBIC BASELINE
        # ----------------------------------------------------

        bicubic = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="bicubic",
            align_corners=False,
        )

        # ----------------------------------------------------
        # FEATURE EXTRACTION
        # ----------------------------------------------------

        features = self.input_conv(x)

        skip = features

        # ----------------------------------------------------
        # RESIDUAL LEARNING
        # ----------------------------------------------------

        features = self.residual_blocks(
            features
        )

        features = self.feature_conv(
            features
        )

        features = features + skip

        # ----------------------------------------------------
        # UPSCALE FEATURES
        # ----------------------------------------------------

        features = self.upscale(
            features
        )

        # ----------------------------------------------------
        # PREDICT RESIDUAL
        # ----------------------------------------------------

        residual = self.output_conv(
            features
        )

        # ----------------------------------------------------
        # ADD TO BICUBIC
        # ----------------------------------------------------

        output = bicubic + residual

        return output