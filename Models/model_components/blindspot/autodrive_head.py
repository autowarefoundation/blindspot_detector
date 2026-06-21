import torch
import torch.nn as nn


class AutoDriveHead(nn.Module):
    """
    Regression + classification head for AutoDrive.

    Inputs
    ------
    feature_prev : (B, C, H, W)  — P5 from backbone(image_prev)
    feature_curr : (B, C, H, W)  — P5 from backbone(image_curr)

    Compression path
    ----------------
    cat([feature_prev, feature_curr], dim=1)  → (B, 2C, H, W)  e.g. C=256 → (B, 512, H, W)
    Conv 2C→256 → SiLU → Conv 256→64 → SiLU → Conv 64→2 → SiLU
    flatten                                   → (B, 2·H·W); e.g. H=16, W=32 → (B, 1024)

    Shared trunk
    ------------
    FC1  : Linear(2·p5_h·p5_w, 768) + SiLU + Dropout(0.1)
    FC2  : Linear(768, 512)          + SiLU + Dropout(0.1)

    Task branches
    -------------
    distance_head  : Linear(512, 1) + ReLU   → d_norm ≥ 0  (L1 vs d_norm_gt ∈ [0,1])
                     d_norm = (150 - min(d,150)) / 150  → to_distance_meters = 150*(1-d_norm)

    curvature_head : Linear(512, 1) + Tanh   → ∈ (-1, 1)  (L1 vs curvature in 1/m)

    flag_head      : Linear(512, 1)           → raw logit   (BCEWithLogitsLoss vs {0,1})
                     No activation here — BCEWithLogitsLoss applies sigmoid internally.
    """

    def __init__(self, in_channels: int = 256, p5_h: int = 16, p5_w: int = 32):
        super().__init__()

        concat_c = 2 * in_channels
        self.conv_1 = nn.Conv2d(concat_c, 256, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        self.act = nn.SiLU(inplace=True)

        flat_dim = 2 * p5_h * p5_w
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

        self.distance_head = nn.Sequential(
            nn.Linear(512, 1),
            nn.ReLU(),
        )

        self.curvature_head = nn.Sequential(
            nn.Linear(512, 1),
            nn.Tanh(),
        )

        # Raw logit — no sigmoid here; BCEWithLogitsLoss applies it numerically stably.
        self.flag_head = nn.Linear(512, 1)

    def forward(self, feature_prev: torch.Tensor, feature_curr: torch.Tensor):
        x = torch.cat([feature_prev, feature_curr], dim=1)
        x = self.conv_1(x)
        x = self.act(x)
        x = self.conv_2(x)
        x = self.act(x)
        x = self.conv_3(x)
        x = self.act(x)
        x = x.flatten(1)

        x = self.fc1(x)
        x = self.fc2(x)

        d_norm      = self.distance_head(x)
        curvature   = self.curvature_head(x)
        flag_logits = self.flag_head(x)

        return d_norm, curvature, flag_logits

    @staticmethod
    def to_distance_meters(d_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalised ReLU output → metres.  distance_m = 150 * (1 - d_norm)."""
        return 150.0 * (1.0 - d_norm)
