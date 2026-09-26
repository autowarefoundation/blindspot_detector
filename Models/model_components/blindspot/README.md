# BlindSpot Occupancy Classification Model

BlindSpot is a temporal occupancy classification network used to determine whether a vehicle blindspot region is **FREE** or **OCCUPIED**.
The model operates on two consecutive fisheye images captured from the same camera and uses temporal feature fusion to detect changes in the blindspot region.

## Problem Definition

Given:

- Previous image `(t-1)`
- Current image `(t)`

Predict:

```text
FREE      = 0
OCCUPIED  = 1
```

The model performs binary classification and does not perform object detection or object localization.

## Architecture

The network consists of three major components:

```text
image_prev ─┐
            ▼
    BlindSpotBackbone
            │
        feature_prev ─┐
                      ├──► BlindSpotHead ──► Occupancy Logit
        feature_curr ─┘
            │
    BlindSpotBackbone
            ▲
image_curr ─┘
```

The same `BlindSpotBackbone` instance processes both images, so the two timesteps share all backbone weights. The classification head concatenates the resulting feature maps along the channel dimension and predicts blindspot occupancy.

## Model Components

```text
blindspot_network.py
blindspot_backbone.py
blindspot_head.py
```

## `blindspot_network.py`

### Purpose

Top-level model definition.
This file assembles all BlindSpot components into a single network and defines the forward pass used during training and inference.

### Responsibilities

- Create the shared backbone
- Create the classification head
- Process the previous frame
- Process the current frame
- Produce a raw occupancy logit

### Input

```text
image_prev : (B, 3, 512, 1024)
image_curr : (B, 3, 512, 1024)
```

### Output

```text
occupancy_logit : (B, 1)
```

## `blindspot_backbone.py`

### Purpose

Feature-extraction backbone.
The backbone converts input images into compact semantic representations that contain the scene context required for occupancy reasoning.

### Design Goals

- Lightweight architecture
- Reuse proven AutoDrive design principles
- Produce high-level scene features for temporal fusion

### Input

```text
(B, 3, 512, 1024)
```

### Output

With the default BlindSpot configuration:

```text
(B, 256, 16, 32)
```

### Fixed Input Resolution

The current architecture is configured for an input resolution of:

```text
1024 × 512
```

Several `CTX` modules are instantiated using feature-map dimensions derived from this resolution. The `BlindSpotHead` is also sized for the resulting `16 × 32` P5 feature map.

Changing the input resolution therefore requires corresponding updates to the `CTX` configuration and head dimensions, and may require retraining or reinitializing affected layers.

### Shared Layer Dependency

The backbone relies on shared perception components imported from:

```text
Models.model_components.blindspot.common_layers
```

Required modules:

- `Conv`
- `SPPF`
- `C2PSA`
- `CTX`

These modules must be available for the BlindSpot backbone to import and run successfully.

### Core Building Blocks

The backbone is composed of reusable perception modules:

- `Conv`
- `SPPF`
- `C2PSA`
- `CTX`

These modules progressively increase the receptive field and semantic richness while reducing spatial resolution.

## `blindspot_head.py`

### Purpose

Temporal occupancy classifier.
This module combines features extracted from the previous and current frames and predicts whether the blindspot is occupied.

### Design Goals

- Learn temporal scene changes
- Fuse information from consecutive frames
- Produce occupancy predictions
- Remain computationally lightweight

### Inputs

```text
feature_prev : (B, 256, 16, 32)
feature_curr : (B, 256, 16, 32)
```

The feature maps are concatenated along the channel dimension before being processed by the head.

### Output

```text
occupancy_logit : (B, 1)
```

### Prediction Mapping

```text
FREE      = 0
OCCUPIED  = 1
```

The head returns a single raw occupancy logit. Applying sigmoid converts the logit into the predicted probability of the blindspot being occupied.

The output is intended for:

```python
torch.nn.BCEWithLogitsLoss()
```

## Training

### Labels

```text
FREE      = 0
OCCUPIED  = 1
```

The target supplied to `BCEWithLogitsLoss` must be floating point and have the same shape as the model output:

```text
(B, 1)
```

### Loss Function

```python
torch.nn.BCEWithLogitsLoss()
```

No sigmoid layer is used inside the network because `BCEWithLogitsLoss` applies the sigmoid operation internally.

## Example Usage

```python
import torch

from Models.model_components.blindspot.blindspot_network import BlindSpot

model = BlindSpot()

occupancy_logit = model(
    image_prev,
    image_curr,
)

occupancy_probability = torch.sigmoid(occupancy_logit)
```

A binary class decision requires an externally selected probability threshold. The model does not define that threshold.

Example loss calculation:

```python
target = occupancy.float().unsqueeze(1)
criterion = torch.nn.BCEWithLogitsLoss()
loss = criterion(occupancy_logit, target)
```

## Design Philosophy

BlindSpot v1 is intentionally simple.

Key principles:

- Binary occupancy classification
- Temporal reasoning using consecutive frames
- Shared backbone for both timesteps
- Efficient experimentation and deployment

The model serves as a baseline architecture that can later be extended.