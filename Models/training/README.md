# BlindSpot Training Pipeline

## Overview

This directory contains the training pipeline for the BlindSpot occupancy-classification model.

BlindSpot is a binary classification task that predicts whether the vehicle blindspot is:

```text
FREE      = 0
OCCUPIED  = 1
```

The model consumes two consecutive fisheye camera frames:

```text
image_prev
image_curr
```

and predicts a single blindspot occupancy state.

The training pipeline focuses on:

- Training and validation
- Optimization
- Checkpoint management
- Experiment tracking
- TensorBoard logging

Dataset parsing and metadata generation are handled separately and are not part of this training pipeline.

## Repository Structure

```text
train_blindspot.py
    Main training entry point and epoch loop

blindspot_trainer.py
    Training, validation, checkpointing,
    visualization, and TensorBoard logging
```

The trainer uses the BlindSpot model implementation and `BlindSpotDataset` from their respective model and data-parsing packages.

## Training Workflow

```text
train_metadata.json       val_metadata.json
          │                        │
          ▼                        ▼
  BlindSpotDataset        BlindSpotDataset
          │                        │
          ▼                        ▼
 Training DataLoader      Validation DataLoader
          │                        │
          ▼                        │
    BlindSpot Network              │
          │                        │
          ▼                        │
   BCEWithLogitsLoss               │
          │                        │
          ▼                        │
     Adam Optimizer                │
          │                        │
          └────────► Validation ◄──┘
                         │
                         ▼
                    Checkpointing
```

Validation is performed after every training epoch.

## Model Output

The BlindSpot network produces a single raw occupancy logit:

```text
occupancy_logit : (B, 1)
```

Class mapping:

```text
0 → FREE
1 → OCCUPIED
```

A sigmoid function converts the logit into an occupancy probability:

```python
occupied_probability = torch.sigmoid(occupancy_logit)
```

Predictions are generated using a probability threshold of `0.5`:

```python
prediction = occupied_probability >= 0.5
```

## Loss Function

BlindSpot uses:

```python
torch.nn.BCEWithLogitsLoss()
```

The model returns a raw logit. No sigmoid is applied before the loss because `BCEWithLogitsLoss` performs the sigmoid operation internally in a numerically stable manner.

Occupancy labels are converted to floating-point targets shaped `(B, 1)` before loss calculation.

## Optimizer

Training uses Adam:

```python
torch.optim.Adam(
    trainable_parameters,
    lr=1e-4,
    weight_decay=1e-5,
)
```

Only parameters with `requires_grad=True` are optimized.

## Training Loop

Each training iteration performs:

```text
Load Batch
    ↓
Forward Pass
    ↓
BCEWithLogitsLoss
    ↓
Backward Pass
    ↓
Gradient Clipping
    ↓
Optimizer Update
```

Gradient clipping is applied before every optimizer step:

```text
max_norm = 10.0
```

## Dataset Loading

The training script requires a metadata directory containing:

```text
train_metadata.json
val_metadata.json
```

Datasets are created with:

```python
train_dataset = BlindSpotDataset(metadata_dir / "train_metadata.json")
val_dataset = BlindSpotDataset(metadata_dir / "val_metadata.json")
```

The training DataLoader uses shuffling. The validation DataLoader does not.

Training stops with a `RuntimeError` if either DataLoader contains no batches.

## Validation

Validation runs after every epoch and reports:

- Sample-weighted validation loss
- Sample-weighted occupancy classification accuracy

Sample weighting prevents a smaller final batch from disproportionately affecting the reported averages.

Validation accuracy uses sigmoid probabilities and the `0.5` prediction threshold.

## Checkpointing

Training outputs are written to:

```text
<root>/training/blindspot/<run-name>/
```

Each run contains:

```text
checkpoints/
├── BlindSpot_last.pth
├── BlindSpot_best.pth
└── BlindSpot_epochNNN.pth

tensorboard/
└── TensorBoard event files
```

### Checkpoint Types

#### `BlindSpot_last.pth`

Latest completed training checkpoint. It is overwritten after each epoch.

#### `BlindSpot_best.pth`

Checkpoint with the lowest validation loss.

#### `BlindSpot_epochNNN.pth`

Archive checkpoint saved after each completed epoch.

### Stored State

A full checkpoint stores:

- Model state
- Optimizer state
- Completed epoch
- Global training step
- Best validation loss

The trainer can also load a weights-only BlindSpot checkpoint. In that case, optimizer state and training counters are reset.

### Resume Support

Automatic resume:

```text
--resume
```

loads `BlindSpot_last.pth` from the selected run directory.

Explicit checkpoint loading:

```text
--checkpoint <path>
```

loads the specified BlindSpot checkpoint. If both `--checkpoint` and `--resume` are provided, the explicit checkpoint takes precedence.

If the requested checkpoint is unavailable, training starts from a fresh state. Weights from AutoDrive or AutoSpeed models are not loaded.

## Model Selection

The best model is selected using the lowest validation loss:

```text
lowest validation loss → BlindSpot_best.pth
```

Validation accuracy is reported for monitoring but is not used for checkpoint selection.

## TensorBoard Logging

TensorBoard logs are written to:

```text
<root>/training/blindspot/<run-name>/tensorboard/
```

The training pipeline records:

```text
Loss/train_total
Loss/train_avg_total
Loss/val_total
Metrics/accuracy_%
Metrics/lr
Metrics/grad_norm
Hist/occupancy_logits
Visualization/sample
Visualization/val_sample
```

### Logged Visualizations

The visualization panel displays:

- Predicted occupancy class
- Ground-truth occupancy class
- Confidence score
- Prediction result, correct or incorrect

The displayed image is `image_prev` from the temporal pair.

Start TensorBoard with:

```bash
tensorboard --logdir <root>/training/blindspot/<run-name>/tensorboard
```

The training script also prints the run-specific command when it starts.

## Default Configuration

```text
Epochs               : 50
Batch size           : 16
DataLoader workers   : 2
Learning rate        : 1e-4
Optimizer            : Adam
Weight decay         : 1e-5
Loss                 : BCEWithLogitsLoss
Gradient clip norm   : 10.0
Prediction threshold : 0.5
Validation           : Every epoch
Model selection      : Lowest validation loss
Log interval         : 100 training steps
Visualization interval: 500 training steps
```

## Command-Line Arguments

```text
--root <path>
    Required. Root used for training outputs.

--metadata-dir <path>
    Required. Directory containing train_metadata.json and val_metadata.json.

--run-name <name>
    Run directory name. If omitted, the script creates run001, run002, and so on.

--resume
    Resume from BlindSpot_last.pth in the selected run directory.

--checkpoint <path>
    Load an explicit BlindSpot checkpoint. Overrides --resume.

--epochs <int>
    Number of training epochs. Default: 50.

--batch-size <int>
    Batch size. Default: 16.

--workers <int>
    Number of DataLoader workers. Default: 2.

--log-every <int>
    Log training scalars and histograms every N training steps. Default: 100.

--vis-every <int>
    Save a training visualization every N training steps. Default: 500.
```

## Example Usage

### Start a New Run

```bash
python train_blindspot.py     --root training_output     --metadata-dir metadata/blindspot
```

When `--run-name` is omitted, the script automatically creates a numbered run directory such as `run001`.

### Start a Named Run

```bash
python train_blindspot.py     --root training_output     --metadata-dir metadata/blindspot     --run-name baseline
```

### Resume Training

Use the same root and run name so the script can locate the run's `BlindSpot_last.pth`:

```bash
python train_blindspot.py     --root training_output     --metadata-dir metadata/blindspot     --run-name baseline     --resume
```

### Load an Explicit Checkpoint

```bash
python train_blindspot.py     --root training_output     --metadata-dir metadata/blindspot     --run-name continued     --checkpoint <path-to-checkpoint>
```

`--checkpoint` takes precedence when both `--checkpoint` and `--resume` are provided.

## Design Principles

The BlindSpot training pipeline v1 prioritizes:

- Binary occupancy classification
- Temporal reasoning from consecutive frames
- Sample-weighted validation
- Reliable checkpointing
- Resume support
- Lightweight TensorBoard monitoring
- Independence from AutoDrive and AutoSpeed model weights

The implementation provides a baseline that can be extended with additional metrics, learning-rate scheduling, class-balancing strategies, focal loss, mixed-precision training, or deployment-specific optimizations.