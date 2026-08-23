# BlindSpot

BlindSpot is a temporal occupancy classification network used to determine whether a vehicle blindspot region is **FREE** or **OCCUPIED**.

The model operates on two consecutive fisheye images captured from the same camera and uses temporal feature fusion to detect changes in the blindspot region.

## Problem Definition

Given:

- Previous image (`t-1`)
- Current image (`t`)

Predict:

```text
FREE      = 0
OCCUPIED  = 1
```

The model performs binary classification and does not perform object detection or object localization.

---

# Architecture

The network consists of three major components:

```text
image_prev ─┐
            │
            ▼
      BlindSpotBackbone
            │
         P5_prev

image_curr ─┐
            │
            ▼
      BlindSpotBackbone
            │
         P5_curr

            ▼
       BlindSpotHead
            │
            ▼
    Occupancy Logits
```

The backbone extracts semantic features independently from both images.

The classification head fuses the temporal features and predicts blindspot occupancy.

---

# Model Components

```text
blindspot_network.py
blindspot_backbone.py
blindspot_head.py
```

---

# blindspot_network.py

## Purpose

Top-level model definition.

This file assembles all BlindSpot components into a single network and defines the forward pass used during training and inference.

### Responsibilities

- Create backbone
- Create classification head
- Process previous frame
- Process current frame
- Produce occupancy logits

### Input

```python
image_prev : (B, 3, 512, 1024)
image_curr : (B, 3, 512, 1024)
```

### Output

```python
occupancy_logits : (B, 2)
```

---

# blindspot_backbone.py

## Purpose

Feature extraction backbone.

The backbone converts input images into compact semantic representations that contain scene context required for occupancy reasoning.

### Design Goals

- Lightweight architecture
- Reuse proven AutoSpeed design principles
- Enable future backbone weight transfer
- Produce high-level scene features for temporal fusion

### Input

```python
(B, 3, 512, 1024)
```

### Output

```python
(B, 256, 16, 32)
```

### Fixed Input Resolution
The backbone is designed and trained for an input resolution of:

```text
1024 × 512
```
Several CTX modules are instantiated using feature-map dimensions derived from this resolution. As a result, the learned CTX parameters are tied to the expected spatial dimensions of the backbone feature maps.

Changing the input resolution may therefore require updating the CTX configuration and retraining or reinitializing the affected layers.

### Shared Layer Dependency
The backbone relies on shared perception components implemented in:
 
```text
Models/model_components/common_layers.py
```

Required modules:

```text
Conv
SPPF
C2PSA
CTX
```

These layers are shared with other perception models and must be available in the repository for the BlindSpot backbone to import and run successfully.

### Core Building Blocks

The backbone is composed of reusable perception modules:

- Conv
- SPPF
- C2PSA
- CTX

These modules progressively increase receptive field and semantic richness while reducing spatial resolution.

---

# blindspot_head.py

## Purpose

Temporal occupancy classifier.

This module combines features extracted from the previous and current frames and predicts whether the blindspot is occupied.

### Design Goals

- Learn temporal scene changes
- Fuse information from consecutive frames
- Produce robust occupancy predictions
- Remain computationally lightweight

### Inputs

```python
feature_prev : (B, 256, 16, 32)
feature_curr : (B, 256, 16, 32)
```

### Output

```python
occupancy_logits : (B, 2)
```

### Prediction Mapping

```text
0 → FREE
1 → OCCUPIED
```

The head returns raw logits intended for:

```python
nn.CrossEntropyLoss()
```

---

# Training

## Labels

```text
FREE      = 0
OCCUPIED  = 1
```

Expected label type:

```python
torch.long
```

## Loss Function

```python
nn.CrossEntropyLoss()
```

No softmax layer is used inside the network because the loss function applies it internally.

---

# Example Usage

```python
model = BlindSpot()

occupancy_logits = model(
    image_prev,
    image_curr,
)

prediction = occupancy_logits.argmax(dim=1)
```

---

# Design Philosophy

BlindSpot V1 is intentionally simple.

Key principles:

- Binary occupancy classification
- Temporal reasoning using consecutive frames
- Shared backbone for both timesteps
- Compatibility with AutoSpeed weight initialization
- Efficient experimentation and deployment

The model serves as a baseline architecture that can later be extended.