"""BlindSpot occupancy classification backbone.

Defines :class:`BlindSpotBackbone`, which extracts a compact P5 feature map
from a single fisheye image for downstream temporal fusion in the
BlindSpotHead.
"""

import torch
import torch.nn as nn

from Models.model_components.blindspot.common_layers import (
    Conv,
    SPPF,
    C2PSA,
    CTX,
)


class BlindSpotBackbone(nn.Module):
    """BlindSpot backbone.
    Extracts a compact P5 feature representation from a single fisheye image.

    Input
    -----
    x : (B, 3, 512, 1024)

    Output
    ------
    p5 : (B, 256, 16, 32)

    Notes
    -----
    - Spatial resolution is progressively reduced.
    - Channel depth is progressively increased.
    - P5 is used by the BlindSpotHead for temporal fusion.
    """

    def __init__(self, width, depth, csp):
        """Build the P1-P5 feature extraction stages.

        Args:
            width: Per-stage channel widths indexed from the stem to P5.
            depth: Per-stage repeat counts for the CTX blocks.
            csp: CSP configuration values passed through to the CTX blocks.
        """
        super().__init__()

        # p1/2
        # 512x1024 -> 256x512
        self.p1 = Conv(
            width[0],
            width[1],
            activation=torch.nn.SiLU(),
            k=3,
            s=2,
            p=1,
        )

        # p2/4
        # 256x512 -> 128x256
        self.p2 = nn.Sequential(
            Conv(
                width[1],
                width[2],
                activation=torch.nn.SiLU(),
                k=3,
                s=2,
                p=1,
            ),
            # Context aggregation block
            CTX(
                width[2],
                width[3],
                depth[0],
                csp[0],
                r=2,
                h=128,
                w=256,
            ),
        )

        # p3/8
        # 128x256 -> 64x128
        self.p3 = nn.Sequential(
            Conv(
                width[3],
                width[3],
                activation=torch.nn.SiLU(),
                k=3,
                s=2,
                p=1,
            ),
            CTX(
                width[3],
                width[4],
                depth[1],
                csp[0],
                r=2,
                h=64,
                w=128,
            ),
        )

        # p4/16
        # 64x128 -> 32x64
        self.p4 = nn.Sequential(
            Conv(
                width[4],
                width[4],
                activation=torch.nn.SiLU(),
                k=3,
                s=2,
                p=1,
            ),
            CTX(
                width[4],
                width[4],
                depth[1],
                csp[0],
                r=2,
                h=32,
                w=64,
            ),
        )

        # p5/32
        # 32x64 -> 16x32
        self.p5 = nn.Sequential(
            Conv(
                width[4],
                width[5],
                activation=torch.nn.SiLU(),
                k=3,
                s=2,
                p=1,
            ),
            # Context-aware feature extraction
            CTX(
                width[5],
                width[5],
                depth[1],
                csp[0],
                r=2,
                h=16,
                w=32,
            ),
            # Multi-scale context aggregation
            SPPF(
                width[5],
                width[5],
            ),
            # Attention-based feature refinement
            C2PSA(
                width[5],
                width[5],
            ),
        )

    def forward(self, x):
        """Run the image through all downsampling stages.

        Args:
            x: Input fisheye image batch of shape (B, 3, 512, 1024).

        Returns:
            The P5 feature map of shape (B, 256, 16, 32).
        """
        # Input image
        # x: (B, 3, 512, 1024)
        p1 = self.p1(x)     # (B, 16, 256, 512)
        p2 = self.p2(p1)    # (B, 64, 128, 256)
        p3 = self.p3(p2)    # (B, 128, 64, 128)
        p4 = self.p4(p3)    # (B, 128, 32, 64)
        p5 = self.p5(p4)    # (B, 256, 16, 32)

        # Highest-level feature map used by BlindSpotHead.
        return p5