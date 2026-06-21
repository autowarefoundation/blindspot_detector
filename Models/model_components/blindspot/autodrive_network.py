import torch
import torch.nn as nn
from pathlib import Path

from Models.model_components.autodrive.autodrive_backbone import AutoDriveBackbone
from Models.model_components.autodrive.autodrive_head import AutoDriveHead

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 512

# Fixed backbone hyperparameters — same 'n' variant as AutoSpeed
_WIDTH = [3, 16, 32, 64, 128, 256]
_DEPTH = [1, 1, 1, 1, 1, 1]
_CSP   = [False, True]


class AutoDrive(nn.Module):
    """
    Shared backbone on previous and current frame.
    Head concatenates P5 maps → conv+SiLU → flatten → MLP → (d_norm, curvature, flag_logit).
    """

    def __init__(self):
        super().__init__()
        self.backbone = AutoDriveBackbone(_WIDTH, _DEPTH, _CSP)
        self.head = AutoDriveHead(
            in_channels=_WIDTH[5],
            p5_h=IMAGE_HEIGHT // 32,
            p5_w=IMAGE_WIDTH // 32,
        )

    def forward(self, image_prev, image_curr):
        """Returns (d_norm (B,1), curvature (B,1), flag_logit (B,1))."""
        feature_prev = self.backbone(image_prev)
        feature_curr = self.backbone(image_curr)
        return self.head(feature_prev, feature_curr)

    def load_backbone_from_autospeed(self, autospeed_ckpt_path: str) -> None:
        """
        Transfer backbone weights from a trained AutoSpeed checkpoint.

        AutoSpeed saves its backbone under the prefix 'net.*' inside
        ckpt['model'].state_dict().  AutoDrive's backbone has the same
        architecture (identical 'n' variant), so all 116 keys transfer 1-to-1
        after stripping the 'net.' prefix.

        Args:
            autospeed_ckpt_path: path to autospeed.pt (or last.pt / best.pt)
        """
        path = Path(autospeed_ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"AutoSpeed checkpoint not found: {path}")

        print(f"Loading backbone weights from AutoSpeed checkpoint: {path}")
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

        # AutoSpeed saves {'epoch': N, 'model': <YOLO instance>}
        if isinstance(ckpt, dict) and "model" in ckpt:
            as_sd = ckpt["model"].state_dict()
        else:
            as_sd = ckpt  # bare state dict fallback

        # Strip the 'net.' prefix to align with AutoDrive backbone key names
        backbone_sd = {k[4:]: v for k, v in as_sd.items() if k.startswith("net.")}

        ad_sd = self.backbone.state_dict()
        matched   = {k: v for k, v in backbone_sd.items() if k in ad_sd and ad_sd[k].shape == v.shape}
        missing   = [k for k in ad_sd if k not in backbone_sd]
        mismatched = [k for k in backbone_sd if k in ad_sd and ad_sd[k].shape != backbone_sd[k].shape]

        if missing or mismatched:
            print(f"  WARNING — {len(missing)} missing, {len(mismatched)} shape mismatches")
            for k in missing[:5]:
                print(f"    missing: {k}")
            for k in mismatched[:5]:
                print(f"    mismatch: {k}  AD={ad_sd[k].shape} AS={backbone_sd[k].shape}")

        self.backbone.load_state_dict(matched, strict=False)
        print(f"  Transferred {len(matched)}/{len(ad_sd)} backbone parameters  ✓")
