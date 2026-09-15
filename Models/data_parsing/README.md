# WoodScape BlindSpot Data Pipeline

## Overview

This directory contains the complete data pipeline used by the BlindSpot Occupancy Classification project.

The pipeline converts the raw WoodScape occupancy dataset into a leakage-safe train/validation dataset and provides PyTorch dataset and dataloader utilities for model training.

The design has two goals:

1. Generate temporally aligned blindspot samples from consecutive fisheye images.
2. Prevent train/validation leakage between highly correlated frames.

The pipeline consists of:

```text
parse_woodscape.py
    Raw dataset → metadata files

load_woodscape.py
    Metadata files → PyTorch Dataset/DataLoader
```

---

# End-to-End Data Flow

```text
WoodScape Dataset
        │
        ▼
parse_woodscape.py
        │
        ├── train_metadata.json
        │
        └── val_metadata.json
                │
                ▼
load_woodscape.py
                │
                ▼
BlindSpotDataset
                │
                ▼
PyTorch DataLoader
                │
                ▼
BlindSpot Network
                │
                ▼
CrossEntropyLoss
```

---

# Dataset Structure

The parser expects the following WoodScape directory structure:

```text
WoodScape/
└── data/
    └── all_images/
        ├── left_fisheye/
        │   ├── occupancy_annotations.json
        │   └── images/
        │       ├── 00005_MVL.png
        │       ├── 00005_MVL_prev.png
        │       └── ...
        │
        └── right_fisheye/
            ├── occupancy_annotations.json
            └── images/
                ├── 00005_MVR.png
                ├── 00005_MVR_prev.png
                └── ...
```

---

# Occupancy Labels

WoodScape annotations:

```json
{
    "00005_MVL": true
}
```

are converted into training labels:

```text
False → FREE     → 0
True  → OCCUPIED → 1
```

---

# Temporal Sample Construction

The model learns occupancy using temporal context.

For every current frame:

```text
00005_MVL.png
```

the parser automatically constructs a temporal pair:

```text
00005_MVL_prev.png
00005_MVL.png
```

which becomes:

```json
{
    "timestep_id": "00005",
    "camera": "left",
    "image_prev": ".../00005_MVL_prev.png",
    "image_curr": ".../00005_MVL.png",
    "occupancy": 1
}
```

Only current-frame annotation entries are used to create samples.

Previous-frame annotation entries are intentionally ignored because they provide temporal context rather than prediction targets.

---

# Metadata Format

The parser generates:

```text
parsed_woodscape/
├── train_metadata.json
└── val_metadata.json
```

Each metadata entry contains:

```json
{
    "timestep_id": "00005",
    "camera": "left",
    "image_prev": "...",
    "image_curr": "...",
    "occupancy": 1
}
```

This metadata format serves as the interface between:

```text
parse_woodscape.py
```

and

```text
load_woodscape.py
```

---

# Train / Validation Split Strategy

## Why Group-Based Splitting?

Temporal frames and camera views originating from the same scene are highly correlated.

A naive random split could place:

```text
00005_MVL
```

into training while:

```text
00005_MVR
```

appears in validation.

This would leak scene information across partitions and artificially inflate validation performance.

To prevent this, the dataset uses:

```python
GroupShuffleSplit(
    train_size=0.70,
    random_state=42,
)
```

with:

```text
group = timestep_id
```

---

## Leakage Prevention

The following samples always remain together:

```text
00005_MVL
00005_MVL_prev
00005_MVR
00005_MVR_prev
```

They are assigned entirely to:

```text
Train
```

or

```text
Validation
```

never both.

After splitting, an explicit leakage validation verifies that no timestep appears in both subsets.

---

# Dataset Integrity Validation

The parser performs several automatic quality checks.

## Missing Image Validation

Verifies:

```text
image_prev exists
image_curr exists
```

Invalid samples are skipped.

---

## Duplicate Sample Detection

A sample is uniquely identified by:

```text
timestep_id
camera
image_prev
image_curr
```

Duplicate samples result in a parser failure.

---

## Camera Coverage Analysis

Reports:

```text
Left-only timesteps
Right-only timesteps
Shared timesteps
```

This provides visibility into dataset consistency between cameras.

---

## Leakage Validation

After splitting, the parser verifies that:

```text
train timestep groups
∩
validation timestep groups
=
empty set
```

Any overlap raises an error.

---

## Class Distribution Check

The parser compares occupancy ratios between:

```text
Training Set
Validation Set
```

A warning is generated if the difference exceeds:

```text
5%
```

This helps identify potentially imbalanced splits while allowing training to continue.

---

## Dataset Statistics

The parser reports:

```text
Total samples
Unique timesteps
Training samples
Validation samples
FREE count
OCCUPIED count
Occupancy ratio
```

for visibility into dataset composition.

---

# Image Preprocessing

All images undergo identical preprocessing.

## Original Image

```text
1280 × 966
```

## Crop

```text
Top    : 70 px
Bottom : 256 px
```

Result:

```text
1280 × 640
```

## Resize

```text
1280 × 640
        ↓
1024 × 512
```

Aspect ratio is preserved:

```text
1280 / 640 = 2.0
1024 / 512 = 2.0
```

---

# Transform Policy

The baseline implementation applies:

```python
transforms.ToTensor()
```

only.

No image normalization is performed in V1.

This is intentional to establish a clean baseline before introducing additional preprocessing steps.

---

# Dataset Output

Each sample returned by `BlindSpotDataset`:

```python
{
    "image_prev": image_prev,
    "image_curr": image_curr,
    "occupancy": occupancy,
}
```

Tensor shapes:

```python
image_prev.shape == (3, 512, 1024)
image_curr.shape == (3, 512, 1024)
```

Label format:

```python
occupancy.dtype == torch.long
```

which is compatible with:

```python
nn.CrossEntropyLoss()
```

---

# Example Usage

## Step 1 — Parse Dataset

```bash
python parse_woodscape.py \
    -i WoodScape/data/all_images \
    -o parsed_woodscape
```

Produces:

```text
parsed_woodscape/
├── train_metadata.json
└── val_metadata.json
```

---

## Step 2 — Create DataLoader

```python
from Models.data_parsing.load_woodscape import create_dataloader

train_loader = create_dataloader(
    metadata_file="parsed_woodscape/train_metadata.json",
    batch_size=32,
    shuffle=True,
)
```

---

## Step 3 — Inspect a Batch

```python
batch = next(iter(train_loader))

print(batch["image_prev"].shape)
print(batch["image_curr"].shape)
print(batch["occupancy"].shape)
```

Expected:

```python
torch.Size([32, 3, 512, 1024])
torch.Size([32, 3, 512, 1024])
torch.Size([32])
```

---

## Step 4 — Train BlindSpot Model

```python
for batch in train_loader:

    occupancy_logits = model(
        batch["image_prev"],
        batch["image_curr"],
    )

    loss = criterion(
        occupancy_logits,
        batch["occupancy"],
    )
```

---

# Integration with BlindSpot Model

The dataloader output is designed to connect directly to the BlindSpot network.

Model input:

```python
occupancy_logits = model(
    batch["image_prev"],
    batch["image_curr"],
)
```

Model output:

```python
occupancy_logits.shape == (B, 2)
```

Class mapping:

```text
occupancy_logits[:, 0] → FREE
occupancy_logits[:, 1] → OCCUPIED
```

Training target:

```python
batch["occupancy"]
```

Training loss:

```python
criterion = torch.nn.CrossEntropyLoss()

loss = criterion(
    occupancy_logits,
    batch["occupancy"],
)
```

---

# Design Principles

The pipeline is intentionally designed around:

- Temporal learning from consecutive fisheye frames
- Leakage-safe train/validation evaluation
- Deterministic dataset generation
- Strong dataset integrity validation
- Direct compatibility with BlindSpot V1 training

These principles ensure that model validation reflects true scene generalization rather than memorization of correlated frames. 