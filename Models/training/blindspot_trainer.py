"""BlindSpot trainer.

Training, validation, checkpointing and TensorBoard logging utilities for the
BlindSpot occupancy classification model.
"""

# BlindSpot trainer
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless backend: figures are written to TensorBoard only.

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

# Make the repository root importable when this file is executed directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Models.model_components.blindspot.blindspot_network import BlindSpot


class AverageMeter:
    """Running average over an epoch."""

    def __init__(self):
        """Initialise the meter in a cleared state."""
        self.reset()

    def reset(self):
        """Clear the accumulated sum, count and average."""
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        """Accumulate a value observed ``n`` times and refresh the average.

        Args:
            val: Observed value (already averaged over ``n`` items).
            n: Number of items the value represents, e.g. the batch size.
        """
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class BlindSpotTrainer:
    """
    Trainer for BlindSpot occupancy classification.

    Output:
        Single occupancy logit
    Loss:
        BCEWithLogitsLoss
    Prediction:
        sigmoid(logit) >= 0.5

    Class mapping
    -------------
    0:  FREE
    1:  OCCUPIED

    TensorBoard
    -----------
    Loss/train_total:           Per-step training loss.
    Loss/train_avg_total:       Epoch-averaged training loss.
    Loss/val_total:             Validation loss.
    Metrics/accuracy_%:         Occupancy classification accuracy.
    Metrics/lr:                 Current learning rate.
    Metrics/grad_norm:          Gradient norm before clipping.
    Hist/occupancy_logits:      Distribution of raw occupancy logits
                                produced by the binary occupancy classifier.
    Visualization/sample:       Example blindspot image with prediction
                                and ground-truth occupancy labels.
    """

    def __init__(self, tensorboard_dir: str = "runs"):
        """Set up the device, model, optimizer, loss and logging state.

        Args:
            tensorboard_dir: Directory used by the TensorBoard SummaryWriter.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"BlindSpotTrainer - device: {self.device}")

        self.model = BlindSpot().to(self.device)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # Keep the initial learning rate aligned with the
        # AutoDrive joint-training configuration to minimize
        # behavioural changes while transitioning to BlindSpot.
        self.learning_rate = 1e-4
        self._setup_optimizer()

        # BlindSpot predicts a single raw occupancy logit.
        #
        # sigmoid(logit) -> probability occupied
        #
        # BCEWithLogitsLoss applies the sigmoid
        # operation internally during training.
        self._classification_loss = nn.BCEWithLogitsLoss()

        # Per-step state
        self.loss: torch.Tensor | None = None
        self._grad_norm: float = 0.0

        # Epoch-level running averages
        self.avg_total = AverageMeter()

        # Cached outputs for TensorBoard logging
        self._occupancy_logits_cpu: torch.Tensor | None = None

        # Visualization state
        self._img_prev_vis: torch.Tensor | None = None
        self._occupancy_pred: int = 0
        self._occupancy_gt_val: int = 0
        self._occupancy_confidence: float = 0.0

    # ------------------------------------------------------------------
    # Optimizer / parameter groups
    # ------------------------------------------------------------------
    def _setup_optimizer(self):
        """Build optimizer over trainable parameters only."""
        trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(trainable, lr=self.learning_rate, weight_decay=1e-5)

    def set_learning_rate(self, lr: float):
        """Update the learning rate on the trainer and all parameter groups.

        Args:
            lr: New learning rate to apply.
        """
        self.learning_rate = lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    # ------------------------------------------------------------------
    # Epoch management
    # ------------------------------------------------------------------
    def reset_averages(self):
        """
        Reset epoch-level running statistics.

        BlindSpot currently tracks only the total classification
        loss because occupancy prediction is a single-task problem.
        Additional metrics such as accuracy, precision, recall,
        and F1 can be added later without changing the trainer
        structure.
        """
        self.avg_total.reset()

    # ------------------------------------------------------------------
    # Batch management
    # ------------------------------------------------------------------
    def set_batch(self, batch: dict):
        """
        Move one temporal BlindSpot sample batch to the training device.

        The dataset provides:
            image_prev
            image_curr
            occupancy

        where occupancy:
            0 -> FREE
            1 -> OCCUPIED

        Args:
            batch: Mapping containing the temporal image pair and label.
        """
        self.image_prev = batch["image_prev"].to(self.device)
        self.image_curr = batch["image_curr"].to(self.device)

        # BCEWithLogitsLoss expects float targets shaped like the logits (B, 1).
        self.occupancy_gt = (
            batch["occupancy"]
            .to(
                device=self.device,
                dtype=torch.float32,
            )
            .view(-1, 1)
        )

        # Retain one image and label for TensorBoard
        # visualizations.
        self._img_prev_vis = batch["image_prev"][0]
        self._occupancy_gt_val = int(batch["occupancy"][0].item())

    # ------------------------------------------------------------------
    # Forward + loss
    # ------------------------------------------------------------------
    def run_model(self):
        """
        Run the BlindSpot network and compute the occupancy
        classification loss.
        """
        occupancy_logits = self.model(
            self.image_prev,
            self.image_curr,
        )

        self.loss = self._classification_loss(
            occupancy_logits,
            self.occupancy_gt,
        )

        batch_size = self.image_prev.size(0)
        self.avg_total.update(self.loss.item(), batch_size)

        # Store predictions for TensorBoard logging and
        # lightweight visualization.
        with torch.no_grad():
            self._occupancy_logits_cpu = occupancy_logits.detach().cpu()

            occupied_probabilities = torch.sigmoid(occupancy_logits)
            predictions = (occupied_probabilities >= 0.5).long()

            self._occupancy_pred = int(predictions[0, 0].item())
            probability_occupied = float(occupied_probabilities[0, 0].item())

            # Report confidence in the predicted class, not in OCCUPIED.
            if self._occupancy_pred == 1:
                self._occupancy_confidence = probability_occupied
            else:
                self._occupancy_confidence = 1.0 - probability_occupied

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(
        self,
        batch: dict,
    ) -> tuple:
        """
        Evaluate one validation batch.

        Args:
            batch: Validation batch in the same format as training batches.

        Returns:
            validation loss
            occupancy classification accuracy in percent
        """
        self.set_batch(batch)

        occupancy_logits = self.model(
            self.image_prev,
            self.image_curr,
        )

        loss = self._classification_loss(
            occupancy_logits,
            self.occupancy_gt,
        )

        occupied_probabilities = torch.sigmoid(occupancy_logits)
        predictions = (occupied_probabilities >= 0.5).long()

        accuracy = (
            (predictions == self.occupancy_gt.long()).float().mean().item() * 100.0
        )

        # Retain the final validation sample for TensorBoard
        # histogram logging and visualization.
        self._occupancy_logits_cpu = occupancy_logits.detach().cpu()

        self._occupancy_pred = int(predictions[0, 0].item())
        probability_occupied = float(occupied_probabilities[0, 0].item())

        if self._occupancy_pred == 1:
            self._occupancy_confidence = probability_occupied
        else:
            self._occupancy_confidence = 1.0 - probability_occupied

        return (
            loss.item(),
            accuracy,
        )

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------
    def loss_backward(self):
        """Backpropagate the cached loss from the most recent forward pass."""
        self.loss.backward()

    def run_optimizer(self):
        """Clip gradients, apply one optimizer step and clear the gradients."""
        # Track gradient norm before clipping to make
        # optimisation issues easier to diagnose.
        all_params = [p for p in self.model.parameters() if p.grad is not None]

        if all_params:
            self._grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=10.0,
            ).item()
        else:
            self._grad_norm = 0.0

        self.optimizer.step()
        self.optimizer.zero_grad()

    def zero_grad(self):
        """Clear accumulated parameter gradients."""
        self.optimizer.zero_grad()

    def get_loss(self) -> float:
        """Return the most recent training loss as a Python float."""
        return self.loss.item()

    # ------------------------------------------------------------------
    # Mode helpers
    # ------------------------------------------------------------------
    def set_train_mode(self):
        """Switch the model to training mode."""
        self.model.train()

    def set_eval_mode(self):
        """Switch the model to evaluation mode."""
        self.model.eval()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        global_step: int,
        best_val_loss: float,
    ):
        """
        Save a complete BlindSpot training checkpoint.

        The model and optimizer states are saved together so
        training can resume without resetting optimisation
        progress.

        Args:
            path: Destination file path for the checkpoint.
            epoch: Index of the epoch just completed.
            global_step: Global training step counter.
            best_val_loss: Best validation loss observed so far.
        """
        print(f"Saving checkpoint -> {path}")

        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(
        self,
        path: str,
    ) -> tuple[int, int, float]:
        """
        Load a BlindSpot checkpoint.

        Full training checkpoints restore:
            model weights
            optimizer state
            epoch
            global step
            best validation loss

        Weights-only BlindSpot checkpoints restore model weights
        and reset the training counters.

        Args:
            path: Path to the checkpoint file.

        Returns:
            start_epoch
            global_step
            best_val_loss
        """
        print(f"Loading checkpoint <- {path}")

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])

            # Optimizer state is optional: training can still resume without it.
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except (KeyError, ValueError, RuntimeError):
                print(
                    "  Optimizer state not restored "
                    "(missing state or parameter groups changed)."
                )

            start_epoch = checkpoint.get(
                "epoch",
                0,
            )
            global_step = checkpoint.get(
                "global_step",
                0,
            )
            best_val_loss = checkpoint.get(
                "best_val_loss",
                float("inf"),
            )

            print(
                f"  Resuming epoch {start_epoch + 1}, "
                f"step {global_step}, "
                f"best validation loss "
                f"{best_val_loss:.4f}"
            )
        else:
            # A weights-only checkpoint can initialize another
            # BlindSpot training run, but it cannot restore the
            # optimizer or training progress.
            self.model.load_state_dict(checkpoint)

            start_epoch = 0
            global_step = 0
            best_val_loss = float("inf")

            print(
                "  Weights-only BlindSpot checkpoint - "
                "training counters reset."
            )

        return (
            start_epoch,
            global_step,
            best_val_loss,
        )

    def save_model(
        self,
        path: str,
    ):
        """
        Save BlindSpot model weights without optimizer or
        training-progress state.

        Args:
            path: Destination file path for the weights.
        """
        torch.save(
            self.model.state_dict(),
            path,
        )

    # ------------------------------------------------------------------
    # TensorBoard - per step
    # ------------------------------------------------------------------
    def log_train_step(
        self,
        step: int,
    ):
        """
        Per-step training diagnostics.

        The BlindSpot baseline tracks only the total
        classification loss and gradient norm.

        Args:
            step: Global training step used as the TensorBoard x-axis.
        """
        self.writer.add_scalar(
            "Loss/train_total",
            self.get_loss(),
            step,
        )
        self.writer.add_scalar(
            "Metrics/grad_norm",
            self._grad_norm,
            step,
        )

    def log_histograms(
        self,
        step: int,
    ):
        """
        Output-distribution histograms.

        The BlindSpot classifier predicts a single
        occupancy logit per sample.

        Logging raw occupancy logits helps identify:
        - collapsed classifiers
        - overconfident predictions
        - poor class separation

        Positive logits indicate OCCUPIED,
        negative logits indicate FREE.

        Args:
            step: Global training step used as the TensorBoard x-axis.
        """
        if self._occupancy_logits_cpu is not None:
            self.writer.add_histogram(
                "Hist/occupancy_logits",
                self._occupancy_logits_cpu,
                step,
            )

    # ------------------------------------------------------------------
    # TensorBoard - per epoch
    # ------------------------------------------------------------------
    def log_train_epoch(
        self,
        epoch: int,
    ):
        """
        Log epoch-level training statistics.

        Args:
            epoch: Index of the completed training epoch.
        """
        self.writer.add_scalar(
            "Loss/train_avg_total",
            self.avg_total.avg,
            epoch,
        )
        self.writer.add_scalar(
            "Metrics/lr",
            self.learning_rate,
            epoch,
        )
        self.writer.flush()

    def log_val_epoch(
        self,
        val_loss: float,
        accuracy: float,
        epoch: int,
    ):
        """
        Log validation metrics for BlindSpot occupancy
        classification.

        Args:
            val_loss: Validation loss for the epoch.
            accuracy: Validation accuracy in percent.
            epoch: Index of the completed epoch.
        """
        self.writer.add_scalar(
            "Loss/val_total",
            val_loss,
            epoch,
        )
        self.writer.add_scalar(
            "Metrics/accuracy_%",
            accuracy,
            epoch,
        )
        self.writer.flush()

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def save_visualization(
        self,
        step: int,
        split: str = "train",
    ):
        """
        Write one annotated BlindSpot sample to TensorBoard.

        The visualization shows:
            predicted occupancy
            ground-truth occupancy
            confidence of the predicted class

        The displayed frame is the previous image from the
        temporal pair. It provides scene context for inspecting
        the model prediction without altering the training data.

        Args:
            step: Global step used as the TensorBoard x-axis.
            split: Either "train" or "val"; selects the TensorBoard tag.
        """
        if self._img_prev_vis is None:
            return

        class_names = (
            "FREE",
            "OCCUPIED",
        )

        predicted_label = class_names[self._occupancy_pred]
        ground_truth_label = class_names[self._occupancy_gt_val]

        prediction_is_correct = self._occupancy_pred == self._occupancy_gt_val

        status = "CORRECT" if prediction_is_correct else "INCORRECT"
        status_color = "lime" if prediction_is_correct else "red"

        # WoodScape images currently use ToTensor() only.
        # No inverse normalization is required before display.
        image = self._img_prev_vis.detach().cpu().permute(1, 2, 0).numpy()

        figure, axis = plt.subplots(
            1,
            1,
            figsize=(12, 6),
        )
        figure.patch.set_facecolor("#1e1e1e")

        axis.imshow(image)
        axis.set_title(
            f"BlindSpot - {split} - step {step}",
            color="white",
            fontsize=10,
        )
        axis.axis("off")

        annotation = "\n".join(
            [
                f"Prediction : {predicted_label}",
                f"Ground truth: {ground_truth_label}",
                f"Confidence : {self._occupancy_confidence:.3f}",
                f"Result     : {status}",
            ]
        )

        axis.text(
            10,
            30,
            annotation,
            color=status_color,
            fontsize=9,
            family="monospace",
            verticalalignment="top",
            bbox={
                "facecolor": "black",
                "alpha": 0.70,
                "edgecolor": "none",
                "pad": 5,
            },
        )

        plt.tight_layout()

        # Separate tags keep training and validation samples distinguishable.
        tag = "Visualization/val_sample" if split == "val" else "Visualization/sample"

        self.writer.add_figure(
            tag,
            figure,
            global_step=step,
        )

        # Release the figure to avoid leaking matplotlib resources.
        plt.close(figure)

    def cleanup(self):
        """Flush and close the TensorBoard writer at the end of training."""
        self.writer.flush()
        self.writer.close()

        print("BlindSpotTrainer: finished.")