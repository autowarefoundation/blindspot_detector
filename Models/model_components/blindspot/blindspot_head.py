"""BlindSpot occupancy classification head.

Defines :class:`BlindSpotHead`, a lightweight head that consumes a pair of
backbone feature maps from consecutive frames and predicts whether the
blindspot region is occupied.
"""

import torch
import torch.nn as nn


class BlindSpotHead(nn.Module):
    """
    Temporal blindspot occupancy classification head.

    Inputs
    ------
    feature_prev : (B, C, H, W) -- P5 features from backbone(image_prev)
    feature_curr : (B, C, H, W) -- P5 features from backbone(image_curr)

    Fusion path
    -----------
    cat([feature_prev, feature_curr], dim=1) -> (B, 2C, H, W)

    Conv 2C->256 -> SiLU
    Conv 256->64 -> SiLU
    Conv 64->2 -> SiLU

    flatten -> (B, 2 * H * W)
               e.g. H=16, W=32 -> (B, 1024)

    Shared trunk
    ------------
    FC1 : Linear(2 * p5_h * p5_w, 768) + SiLU + Dropout(0.1)
    FC2 : Linear(768, 512) + SiLU + Dropout(0.1)

    Classification branch
    ---------------------
    occupancy_head : Linear(512, 1) -> raw occupancy logit (B, 1)

        (BCEWithLogitsLoss vs {FREE=0, OCCUPIED=1})

        sigmoid(logit) -> probability occupied

    No sigmoid here. BCEWithLogitsLoss applies the sigmoid operation
    internally during training.

    Output
    ------
    occupancy_logits : (B, 1)
    """

    def __init__(self, in_channels: int = 256, p5_h: int = 16, p5_w: int = 32):
        """
        Build the fusion stack, shared trunk and occupancy classifier.

        Args:
            in_channels: Channel count of each incoming P5 feature map.
            p5_h: Spatial height of the P5 feature map.
            p5_w: Spatial width of the P5 feature map.
        """
        super().__init__()

        # Temporal fusion doubles the channel count:
        # (B,C,H,W) + (B,C,H,W) -> (B,2C,H,W)
        concat_c = 2 * in_channels

        # Convolutional fusion stack.
        # Learns temporal and spatial relationships between the
        # previous and current feature maps.
        self.conv_1 = nn.Conv2d(concat_c, 256, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)

        self.act = nn.SiLU(inplace=True)

        # After Conv3:
        # (B,2,16,32) -> flatten -> (B,1024)
        flat_dim = 2 * p5_h * p5_w

        # Shared feature reasoning trunk.
        # Produces a compact scene representation used by the
        # classification branch.
        self.fc1 = nn.Sequential(
            nn.Linear(flat_dim, 768),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.1),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(768, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.1),
        )

        # Final occupancy classifier.
        #
        # sigmoid(logit) -> probability occupied
        #
        # BCEWithLogitsLoss applies the sigmoid operation
        # internally during training.
        self.occupancy_head = nn.Linear(512, 1)

    def forward(
        self,
        feature_prev: torch.Tensor,
        feature_curr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the raw blindspot occupancy logit for a frame pair.

        Args:
            feature_prev: P5 features of the previous frame, shape (B, C, H, W).
            feature_curr: P5 features of the current frame, shape (B, C, H, W).

        Returns:
            Raw occupancy logits of shape (B, 1); no sigmoid is applied.
        """
        # Concatenate temporal features from t-1 and t.
        # (B,256,16,32) -> (B,512,16,32)
        x = torch.cat([feature_prev, feature_curr], dim=1)

        # Temporal feature fusion and compression.
        x = self.conv_1(x)
        x = self.act(x)

        x = self.conv_2(x)
        x = self.act(x)

        x = self.conv_3(x)
        x = self.act(x)

        # Convert feature maps into a feature vector.
        # (B,2,16,32) -> (B,1024)
        x = x.flatten(1)

        # Shared reasoning layers.
        x = self.fc1(x)
        x = self.fc2(x)

        # Raw blindspot occupancy logit.
        # Shape: (B,1)
        occupancy_logits = self.occupancy_head(x)

        return occupancy_logits