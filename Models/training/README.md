# BlindSpot Training Pipeline

## Overview

This directory contains the training framework used by the BlindSpot occupancy-classification model.

BlindSpot is a binary classification task that predicts whether a vehicle blindspot is:

```text
FREE      = 0
OCCUPIED  = 1
```

Unlike object-detection models, BlindSpot predicts a single occupancy state. The training pipeline is therefore focused on binary classification, optimization stability, validation, checkpointing, and experiment monitoring.

## Repository Structure

```text
train_blindspot.py
    Main training entry point and epoch loop

blindspot_trainer.py
    Model training, validation, checkpointing, and TensorBoard utilities
```

## Training Workflow

```text
WoodScape Dataset
        │
        ▼
LoadDataBlindSpot
        │
        ├── Training DataLoader
        └── Validation DataLoader
                │
                ▼
        BlindSpot Network
                │
                ▼
       BCEWithLogitsLoss
                │
                ▼
          Adam Optimizer
                │
                ▼
           Validation
                │
                ▼
          Checkpointing
```

Validation is performed after every training epoch.

## `train_blindspot.py`

### Purpose

`train_blindspot.py` is the main entry point for BlindSpot training and validation. It creates the output directories, loads the dataset, builds the dataloaders, manages checkpoint loading, and runs the training and validation loops.

### Responsibilities

#### Dataset Loading

The training script creates the dataset through:

```python
LoadDataBlindSpot(args.root)
```

It expects the returned object to provide:

```text
data.train
data.val
```

The training loader uses shuffling, while the validation loader does not.

#### Training

During each training iteration:

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

Gradients are clipped to a maximum norm of `10.0` before each optimizer step.

#### Validation

Validation is executed after every epoch. The validation loop computes sample-weighted averages for:

- Validation loss
- Occupancy classification accuracy

A probability threshold of `0.5` is used to convert sigmoid probabilities into binary predictions.

#### Checkpointing

Training outputs are written under:

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

Checkpoint roles:

- `BlindSpot_last.pth`: latest completed epoch
- `BlindSpot_best.pth`: checkpoint with the lowest validation loss
- `BlindSpot_epochNNN.pth`: checkpoint saved after each completed epoch

A full checkpoint stores:

- Model state
- Optimizer state
- Completed epoch
- Global training step
- Best validation loss

The trainer can also load a weights-only BlindSpot checkpoint. In that case, optimizer state and training counters are reset.

### Resume Options

An explicit checkpoint takes precedence over automatic resume:

```text
--checkpoint <path>
```

Automatic resume loads `BlindSpot_last.pth` from the selected run directory:

```text
--resume
```

If the requested checkpoint is unavailable, training starts from a fresh state.

Weights from AutoDrive or AutoSpeed models are not loaded by this training pipeline.

## `blindspot_trainer.py`

### Purpose

`blindspot_trainer.py` contains the training, validation, optimization, checkpointing, and TensorBoard logic used by the main training script.

### Model and Device

The trainer creates the `BlindSpot` model and selects the runtime device automatically:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Batch Preparation

Each batch contains:

```text
image_prev
image_curr
occupancy
```

Occupancy labels are converted to floating-point targets shaped `(B, 1)` for binary-logit training.

### Loss Function

BlindSpot is trained using:

```python
torch.nn.BCEWithLogitsLoss()
```

The model produces a single raw occupancy logit:

```text
occupancy_logit : (B, 1)
```

`BCEWithLogitsLoss` applies the sigmoid operation internally. Sigmoid is therefore not applied before calculating the training loss.

### Optimizer

BlindSpot uses the Adam optimizer:

```python
torch.optim.Adam(
    trainable_parameters,
    lr=1e-4,
    weight_decay=1e-5,
)
```

Only parameters with `requires_grad=True` are passed to the optimizer.

### Prediction

During validation and visualization:

```python
occupied_probability = torch.sigmoid(occupancy_logit)
prediction = occupied_probability >= 0.5
```

Class mapping:

```text
0 → FREE
1 → OCCUPIED
```

## Model Selection

The best model is selected using the lowest validation loss:

```text
best validation loss → BlindSpot_best.pth
```

Validation accuracy is reported for monitoring but is not used for best-checkpoint selection.

## TensorBoard Logging

TensorBoard event files are written to:

```text
<root>/training/blindspot/<run-name>/tensorboard/
```

The trainer records:

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

The visualizations show:

- Predicted occupancy class
- Ground-truth occupancy class
- Confidence in the predicted class
- Whether the prediction is correct

The displayed image is the previous frame from the temporal pair.

Start TensorBoard with the run-specific path printed by the training script:

```bash
tensorboard --logdir <root>/training/blindspot/<run-name>/tensorboard
```

## Training Configuration

Default command-line values:

```text
Epochs              : 50
Batch size          : 16
DataLoader workers  : 2
Learning rate       : 1e-4
Optimizer           : Adam
Weight decay        : 1e-5
Loss                : BCEWithLogitsLoss
Gradient clip norm  : 10.0
Prediction threshold: 0.5
Validation          : Every epoch
Model selection     : Lowest validation loss
```

### Command-Line Arguments

```text
--root <path>          Required dataset root and training-output root
--run-name <name>      Run directory name
--resume               Resume from BlindSpot_last.pth
--checkpoint <path>    Load an explicit BlindSpot checkpoint
--epochs <int>         Number of epochs, default: 50
--batch-size <int>     Batch size, default: 16
--workers <int>        DataLoader workers, default: 2
--log-every <int>      Scalar and histogram interval, default: 100 steps
--vis-every <int>      Training visualization interval, default: 500 steps
```

When `--run-name` is omitted, the script creates an automatically numbered directory such as `run001`, `run002`, or `run003`.

## Example Training Runs

### Start a New Run

```bash
python train_blindspot.py     --root <dataset-root>
```

### Use a Named Run

```bash
python train_blindspot.py     --root <dataset-root>     --run-name baseline
```

### Resume a Run

Use the same root and run name so the script can locate that run's `BlindSpot_last.pth`:

```bash
python train_blindspot.py     --root <dataset-root>     --run-name baseline     --resume
```

### Load an Explicit Checkpoint

```bash
python train_blindspot.py     --root <dataset-root>     --run-name continued     --checkpoint <path-to-checkpoint>
```

`--checkpoint` takes precedence when both `--checkpoint` and `--resume` are provided.

## Design Principles

BlindSpot Training Pipeline v1 prioritizes:

- Simple binary occupancy classification
- Temporal reasoning from consecutive frames
- Sample-weighted validation
- Reliable checkpointing and resume support
- Lightweight TensorBoard monitoring
- A focused training pipeline without object-detection utilities

The training pipeline provides a baseline that can be extended with additional metrics, scheduling, class-balancing strategies, or other optimization techniques.