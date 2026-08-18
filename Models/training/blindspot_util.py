"""
blindspot_util.py

Utility functions and helper classes used by the BlindSpot
occupancy-classification training pipeline.

This module contains:

    - Reproducibility utilities
    - Multi-processing configuration
    - Learning-rate schedulers
    - Exponential Moving Average (EMA)
    - Classification metrics
    - Optimizer helper functions
    - Checkpoint utilities

The BlindSpot task is a binary classification problem:

    0 -> FREE
    1 -> OCCUPIED

The utilities in this file are intentionally kept lightweight.
Unlike AutoSpeed, BlindSpot does not require:

    - Anchors
    - IoU computation
    - NMS
    - Object detection losses
    - mAP calculation
"""

import copy
import math
import random

import numpy as np
import torch


def setup_seed(seed=0):
    """
    Configure seeds for reproducible experiments.

    This allows fair comparison between:

        - Model architectures
        - Hyperparameter choices
        - Training strategies

    without training randomness becoming a confounding factor.

    Note:
        Perfect reproducibility across different hardware and
        software versions is not guaranteed, but this greatly
        reduces experiment-to-experiment variability.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Configure the runtime environment for efficient training.

    Main goals:

        1. Prevent OpenCV from creating excessive threads.
        2. Reduce CPU oversubscription.
        3. Improve DataLoader stability.
        4. Standardize multiprocessing behavior.

    These settings are inherited from the AutoSpeed and
    AutoSteer training infrastructure.
    """

    import cv2
    from os import environ
    from platform import system

    # Use fork on Linux to reduce DataLoader startup overhead.
    if system() != "Windows":
        torch.multiprocessing.set_start_method(
            "fork",
            force=True,
        )

    # Disable OpenCV internal multithreading.
    # Multiple DataLoader workers + OpenCV threads can
    # easily oversubscribe the CPU.
    cv2.setNumThreads(0)

    if "OMP_NUM_THREADS" not in environ:
        environ["OMP_NUM_THREADS"] = "1"

    if "MKL_NUM_THREADS" not in environ:
        environ["MKL_NUM_THREADS"] = "1"


class AverageMeter:
    """
    Tracks running averages during training and validation.

    Example usage:

        - Training loss
        - Validation loss
        - Accuracy
        - F1 score

    Instead of storing every individual value, the class
    maintains:

        Running Sum
        Sample Count
        Average
    """

    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(
        self,
        value,
        n,
    ):
        """
        Update running statistics.

        Args:
            value:
                Metric value to accumulate.

            n:
                Number of samples represented by value.
        """

        if not math.isnan(float(value)):
            self.num += n
            self.sum += value * n
            self.avg = self.sum / self.num


class LinearLR:
    """
    Learning-rate scheduler.

    Schedule:

        Warmup
            ↓
        Linear Decay

    Warmup helps prevent unstable optimization during the
    first training iterations when model weights are still
    adapting to the dataset.

    After warmup, the learning rate decays linearly toward
    the configured minimum learning rate.
    """

    def __init__(
        self,
        args,
        params,
        num_steps,
    ):
        max_lr = params["max_lr"]
        min_lr = params["min_lr"]

        warmup_steps = int(
            params["warmup_epochs"] * num_steps
        )

        decay_steps = int(
            args.epochs * num_steps -
            warmup_steps
        )

        warmup_lr = np.linspace(
            min_lr,
            max_lr,
            int(warmup_steps),
            endpoint=False,
        )

        decay_lr = np.linspace(
            max_lr,
            min_lr,
            decay_steps,
        )

        self.total_lr = np.concatenate(
            (
                warmup_lr,
                decay_lr,
            )
        )

    def step(
        self,
        step,
        optimizer,
    ):
        """
        Update optimizer learning rate for current step.
        """

        for param_group in optimizer.param_groups:
            param_group["lr"] = self.total_lr[step]


class CosineLR:
    """
    Learning-rate scheduler.

    Schedule:

        Warmup
            ↓
        Cosine Decay

    Cosine decay gradually reduces the learning rate
    following a smooth cosine curve.

    Sometimes provides slightly better convergence than
    linear decay, but LinearLR is recommended for the
    initial BlindSpot baseline.
    """

    def __init__(
        self,
        args,
        params,
        num_steps,
    ):
        max_lr = params["max_lr"]
        min_lr = params["min_lr"]

        warmup_steps = int(
            max(
                params["warmup_epochs"] * num_steps,
                100,
            )
        )

        decay_steps = int(
            args.epochs * num_steps -
            warmup_steps
        )

        warmup_lr = np.linspace(
            min_lr,
            max_lr,
            warmup_steps,
        )

        decay_lr = []

        for step in range(
            1,
            decay_steps + 1,
        ):
            alpha = math.cos(
                math.pi *
                step /
                decay_steps
            )

            decay_lr.append(
                min_lr +
                0.5 *
                (max_lr - min_lr) *
                (1 + alpha)
            )

        self.total_lr = np.concatenate(
            (
                warmup_lr,
                decay_lr,
            )
        )

    def step(
        self,
        step,
        optimizer,
    ):
        """
        Update optimizer learning rate for current step.
        """

        for param_group in optimizer.param_groups:
            param_group["lr"] = self.total_lr[step]


class EMA:
    """
    Exponential Moving Average (EMA) of model weights.

    During training, model weights can fluctuate because of
    stochastic gradient updates.

    EMA maintains a smoothed version of the weights:

        ema = decay * ema
            + (1 - decay) * current_model

    Validation and checkpointing are typically performed
    using EMA weights because they often produce:

        - More stable validation metrics
        - Better generalization
        - Less noisy model selection
    """

    def __init__(
        self,
        model,
        decay=0.9999,
        tau=2000,
        updates=0,
    ):
        self.ema = copy.deepcopy(
            model
        ).eval()

        self.updates = updates

        # Exponential ramp-up of decay.
        # Helps EMA adapt faster during early training.
        self.decay = (
            lambda x:
            decay *
            (1 - math.exp(-x / tau))
        )

        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    def update(
        self,
        model,
    ):
        """
        Update EMA parameters after each optimizer step.
        """

        if hasattr(model, "module"):
            model = model.module

        with torch.no_grad():

            self.updates += 1

            d = self.decay(
                self.updates
            )

            model_state = model.state_dict()

            for key, value in self.ema.state_dict().items():

                if value.dtype.is_floating_point:

                    value *= d
                    value += (
                        1 - d
                    ) * model_state[key].detach()


def set_params(
    model,
    weight_decay,
):
    """
    Create optimizer parameter groups.

    Weight decay should generally be applied only to
    trainable weights that benefit from regularization.

    No weight decay:

        - Bias parameters
        - Normalization layers

    Apply weight decay:

        - Convolution weights
        - Linear weights

    This follows standard computer-vision training practice.
    """

    p_no_decay = []
    p_decay = []

    norm_layers = tuple(
        value
        for key, value
        in torch.nn.__dict__.items()
        if "Norm" in key
    )

    for module in model.modules():

        for name, parameter in module.named_parameters(
            recurse=False
        ):

            if not parameter.requires_grad:
                continue

            if name == "bias":

                p_no_decay.append(
                    parameter
                )

            elif (
                name == "weight"
                and isinstance(
                    module,
                    norm_layers,
                )
            ):

                p_no_decay.append(
                    parameter
                )

            else:

                p_decay.append(
                    parameter
                )

    return [
        {
            "params": p_no_decay,
            "weight_decay": 0.0,
        },
        {
            "params": p_decay,
            "weight_decay": weight_decay,
        },
    ]


def strip_optimizer(
    checkpoint_file,
):
    """
    Prepare a training checkpoint for deployment or inference.

    Actions:

        - Convert model to FP16
        - Disable gradients
        - Save compact checkpoint

    This reduces model size and memory usage.
    """

    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint["model"].half()

    for parameter in checkpoint["model"].parameters():
        parameter.requires_grad = False

    torch.save(
        checkpoint,
        checkpoint_file,
    )


def compute_confusion_matrix(
    y_true,
    y_pred,
):
    """
    Compute confusion-matrix elements for BlindSpot
    occupancy classification.

    Class mapping:

        0 -> FREE
        1 -> OCCUPIED

    Returns:

        TP:
            Occupied correctly predicted as occupied.

        TN:
            Free correctly predicted as free.

        FP:
            Free incorrectly predicted as occupied.

        FN:
            Occupied incorrectly predicted as free.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # True Positives:
    # Occupied blindspots correctly classified as occupied.
    tp = np.sum(
        (y_true == 1) &
        (y_pred == 1)
    )

    # True Negatives:
    # Free blindspots correctly classified as free.
    tn = np.sum(
        (y_true == 0) &
        (y_pred == 0)
    )

    # False Positives:
    # Free blindspots incorrectly classified as occupied.
    # These generate nuisance warnings.
    fp = np.sum(
        (y_true == 0) &
        (y_pred == 1)
    )

    # False Negatives:
    # Occupied blindspots incorrectly classified as free.
    # These are typically the most safety-relevant failures.
    fn = np.sum(
        (y_true == 1) &
        (y_pred == 0)
    )

    return tp, tn, fp, fn


def compute_classification_metrics(
    y_true,
    y_pred,
):
    """
    Compute BlindSpot classification metrics.

    Metrics:

        Accuracy
            Overall prediction correctness.

        Precision
            Percentage of OCCUPIED predictions that
            are actually occupied.

        Recall
            Percentage of real OCCUPIED samples that
            are successfully detected.

        F1 Score
            Harmonic mean of precision and recall.

    F1 Score is used as the primary model-selection
    metric because it balances:

        Precision
        Recall

    and provides a better picture than accuracy when
    class distributions are not perfectly balanced.
    """

    tp, tn, fp, fn = compute_confusion_matrix(
        y_true,
        y_pred,
    )

    accuracy = (
        (tp + tn)
        / max(
            tp + tn + fp + fn,
            1,
        )
    )

    # Precision answers:
    #
    # "When the model predicts OCCUPIED,
    # how often is it correct?"
    precision = (
        tp
        / max(
            tp + fp,
            1,
        )
    )

    # Recall answers:
    #
    # "How many true OCCUPIED blindspots
    # did the model successfully detect?"
    #
    # This is a particularly important metric from
    # a safety perspective because low recall implies
    # missed occupied blindspots.
    recall = (
        tp
        / max(
            tp + fn,
            1,
        )
    )

    # Harmonic mean of precision and recall.
    #
    # Used as the best-model criterion when deciding
    # whether to update best.pt.
    f1 = (
        2 *
        precision *
        recall
        / max(
            precision + recall,
            1e-12,
        )
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }