import torch
import torch.nn as nn


class BlindSpotHead(nn.Module):
    """
    Temporal blindspot occupancy classification head.

    Inputs
    ------
    feature_prev : (B, C, H, W)  -- P5 features from backbone(image_prev)
    feature_curr : (B, C, H, W)  -- P5 features from backbone(image_curr)

    Fusion path
    -----------
    cat([feature_prev, feature_curr], dim=1) -> (B, 2C, H, W)

    Conv 2C->256 -> SiLU
    Conv 256->64 -> SiLU
    Conv 64->2   -> SiLU

    flatten -> (B, 2*H*W);
                e.g. H=16, W=32 -> (B, 1024)

    Shared trunk
    ------------
    FC1 : Linear(2*p5_h*p5_w, 768) + SiLU + Dropout(0.1)
    FC2 : Linear(768, 512)         + SiLU + Dropout(0.1)

    Classification branch
    ---------------------
    occupancy_head : Linear(512, 2) -> raw logits (B, 2)
                    (CrossEntropyLoss vs {FREE=0, OCCUPIED=1})

                    logits[:, 0] -> FREE
                    logits[:, 1] -> OCCUPIED

                    No softmax here — CrossEntropyLoss applies it internally.

    Output
    ------
    occupancy_logits : (B, 2)
    """

    def __init__(self,
                 in_channels: int = 256,
                 p5_h: int = 16,
                 p5_w: int = 32):
        super().__init__()

        # Temporal fusion doubles the channel count:
        # (B,C,H,W) + (B,C,H,W) -> (B,2C,H,W)
        concat_c = 2 * in_channels

        # Convolutional fusion stack.
        # Learns temporal and spatial relationships between the previous and current feature maps.
        self.conv_1 = nn.Conv2d(
            concat_c, 256,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv_2 = nn.Conv2d(
            256, 64,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv_3 = nn.Conv2d(
            64, 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.act = nn.SiLU(inplace=True)

        # After Conv3:
        # (B,2,16,32) -> flatten -> (B,1024)
        flat_dim = 2 * p5_h * p5_w

        # Shared feature reasoning trunk.
        # Produces a compact scene representation used by the classification branch.
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

        # Final occupancy classifier:
        #   output[:,0] -> FREE
        #   output[:,1] -> OCCUPIED
        # Raw occupancy logits — no softmax here;
        # CrossEntropyLoss applies softmax internally in a numerically stable manner.

        self.occupancy_head = nn.Linear(512, 2)

    def forward(
            self,
            feature_prev: torch.Tensor,
            feature_curr: torch.Tensor
    ) -> torch.Tensor:

        # Concatenate temporal features from t-1 and t
        # (B,256,16,32) -> (B,512,16,32)
        x = torch.cat(
            [feature_prev, feature_curr],
            dim=1,
        )

        # Temporal feature fusion and compression
        x = self.conv_1(x)
        x = self.act(x)

        x = self.conv_2(x)
        x = self.act(x)

        x = self.conv_3(x)
        x = self.act(x)

        # Convert feature maps into a feature vector
        # (B,2,16,32) -> (B,1024)
        x = x.flatten(1)

        # Shared reasoning layers
        x = self.fc1(x)
        x = self.fc2(x)

        # Raw classification blindspot occupnacy logits
        # Shape: (B,2)
        occupancy_logits = self.occupancy_head(x)

        return occupancy_logits