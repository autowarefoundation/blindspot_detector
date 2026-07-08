import torch
import torch.nn as nn
from pathlib import Path

from Models.model_components.blindspot.blindspot_backbone import BlindSpotBackbone
from Models.model_components.blindspot.blindspot_head import BlindSpotHead


IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 512

# Fixed backbone hyperparameters — same 'n' variant as AutoSpeed
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

    def load_backbone_from_autospeed(self, autospeed_ckpt_path: str) -> None:
        """
        Transfer backbone weights from a trained AutoSpeed checkpoint.

        AutoSpeed saves its backbone under the prefix 'net.*' inside
        ckpt['model'].state_dict(). BlindSpotBackbone has the same
        architecture (identical 'n' variant), so all 116 keys
        transfer 1-to-1 after stripping the 'net.' prefix.

        Args:
            autospeed_ckpt_path: path to autospeed.pt (or last.pt / best.pt)
        """

        # Validate checkpoint path
        path = Path(autospeed_ckpt_path)

        if not path.exists():
            raise FileNotFoundError(
                f"AutoSpeed checkpoint not found: {path}"
            )

        print(
            f"Loading backbone weights from AutoSpeed checkpoint: {path}"
        )

        # Load checkpoint
        ckpt = torch.load(
            str(path),
            map_location="cpu",
            weights_only=False,
        )
   
        # AutoSpeed saves:
        # {
        #     "epoch": N,
        #     "model": <YOLO instance>
        # }
        # Extract the underlying state_dict. Fallback to the checkpoint itself if it is already a state_dict.

        if isinstance(ckpt, dict) and "model" in ckpt:
            autospeed_sd = ckpt["model"].state_dict()
        else:
            autospeed_sd = ckpt

        # Convert AutoSpeed key names -> BlindSpotBackbone key names
        # AutoSpeed backbone parameters are stored as 'net.*'
        # Remove the 'net.' prefix so keys match the backbone state_dict() of this model.
        backbone_sd = {
            k[4:]: v
            for k, v in autospeed_sd.items()
            if k.startswith("net.")
        }

        # Current BlindSpot backbone parameters
        current_sd = self.backbone.state_dict()

        # Keep only compatible parameters
        # Transfer parameters that:
        #   1. Exist in both models
        #   2. Have identical tensor shapes
        matched = {
            k: v
            for k, v in backbone_sd.items()
            if k in current_sd
            and current_sd[k].shape == v.shape
        }

        # Backbone parameters that were not found
        missing = [
            k
            for k in current_sd
            if k not in backbone_sd
        ]

        # Parameters with matching names but different tensor shapes
        mismatched = [
            k
            for k in backbone_sd
            if k in current_sd
            and current_sd[k].shape != backbone_sd[k].shape
        ]

        # Report loading statistics
        if missing or mismatched:
            print(
                f"  WARNING — {len(missing)} missing, "
                f"{len(mismatched)} shape mismatches"
            )

            for k in missing[:5]:
                print(f"    missing: {k}")

            for k in mismatched[:5]:
                print(
                    f"    mismatch: {k}  "
                    f"BS={current_sd[k].shape} "
                    f"AS={backbone_sd[k].shape}"
                )

        # Load matching backbone weights
        self.backbone.load_state_dict(
            matched,
            strict=False,
        )

        print(
            f"  Transferred "
            f"{len(matched)}/{len(current_sd)} "
            f"backbone parameters ✓"
        )