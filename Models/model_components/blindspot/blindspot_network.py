import torch
import torch.nn as nn
from pathlib import Path

from Models.model_components.blindspot.blindspot_backbone import BlindSpotBackbone
from Models.model_components.blindspot.blindspot_head import BlindSpotHead


IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 512

# Fixed backbone hyperparameters — same 'n' variant as AutoDrive
_WIDTH = [3, 16, 32, 64, 128, 256]
_DEPTH = [1, 1, 1, 1, 1, 1]
_CSP = [False, True]


class BlindSpot(nn.Module):
    """
    BlindSpot occupancy classification model.

    A shared backbone processes two consecutive frames (t-1, t).
    The resulting P5 feature maps are fused by BlindSpotHead to
    classify blindspot occupancy.

    Inputs
    ------
    image_prev : (B, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    image_curr : (B, 3, IMAGE_HEIGHT, IMAGE_WIDTH)

    Output
    ------
    occupancy_logits : (B, 2)

        occupancy_logits[:, 0] -> FREE
        occupancy_logits[:, 1] -> OCCUPIED
    """

    def __init__(self):
        super().__init__()

        # Shared feature extractor for image(t-1) and image(t)
        self.backbone = BlindSpotBackbone(
            _WIDTH,
            _DEPTH,
            _CSP,
        )

        # BlindSpot occupancy classification head
        self.head = BlindSpotHead(
            in_channels = _WIDTH[5],
            p5_h = IMAGE_HEIGHT // 32,
            p5_w = IMAGE_WIDTH // 32,
        )

    def forward(
            self,
            image_prev: torch.Tensor,
            image_curr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        image_prev -> Backbone -> feature_prev
        image_curr -> Backbone -> feature_curr

        feature_prev + feature_curr
            -> BlindSpotHead
            -> occupancy logits
        """

        # Extract P5 features from the previous frame
        feature_prev = self.backbone(image_prev)

        # Extract P5 features from the current frame
        feature_curr = self.backbone(image_curr)

        # Temporal fusion + occupancy classification
        occupancy_logits = self.head(
            feature_prev,
            feature_curr,
        )

        return occupancy_logits