"""Main training loop for BlindSpot occupancy classification.

The model receives two consecutive WoodScape fisheye images:
    image_prev
    image_curr

and predicts whether the blindspot is:
    FREE      -> 0
    OCCUPIED  -> 1

Training outputs are stored in a run-specific directory:
    checkpoints/BlindSpot_last.pth
        Latest complete training checkpoint.
    checkpoints/BlindSpot_best.pth
        Checkpoint with the lowest validation loss.
    checkpoints/BlindSpot_epochNNN.pth
        Checkpoint saved after each completed epoch.
    tensorboard/
        TensorBoard event files.

The trainer supports:
    fresh BlindSpot training
    resume from BlindSpot_last.pth
    loading an explicit BlindSpot checkpoint

Weights from AutoDrive or AutoSpeed models are not loaded.

TensorBoard records:
    Loss/train_total
    Loss/train_avg_total
    Loss/val_total
    Metrics/accuracy_%
    Metrics/lr
    Metrics/grad_norm
    Hist/occupancy_logits
    Visualization/sample
    Visualization/val_sample
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader

# Make the repository root importable when this file is executed directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Models.data_parsing.load_woodscape import BlindSpotDataset
from Models.training.blindspot_trainer import BlindSpotTrainer


def _run_val(trainer: BlindSpotTrainer, loader: DataLoader):
    """
    Compute sample-weighted validation loss and accuracy.

    Args:
        trainer: Trainer holding the model under evaluation.
        loader: Validation DataLoader.

    Returns:
        average validation loss
        average occupancy accuracy in percent
    """
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0

    # Weight per-batch metrics by batch size so a partial final batch
    # does not skew the averages.
    for batch in loader:
        loss, accuracy = trainer.validate(batch)

        batch_size = batch["occupancy"].size(0)

        total_loss += loss * batch_size
        total_accuracy += accuracy * batch_size
        total_samples += batch_size

    if total_samples == 0:
        return 0.0, 0.0

    return total_loss / total_samples, total_accuracy / total_samples


def main():
    """Parse arguments, build the data pipeline and run the training loop."""
    parser = ArgumentParser()

    parser.add_argument("--root", required=True,
                        help="BlindSpot dataset root. Training outputs are written to {root}/training/blindspot/<run-name>/")
    parser.add_argument("--metadata-dir", required=True,
                        help="Directory containing train_metadata.json and val_metadata.json")
    parser.add_argument("--run-name", default="",
                        help="Sub-folder name for this run (default: auto-numbered run001, run002, ...)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from BlindSpot_last.pth in the run directory")
    parser.add_argument("--checkpoint", default="",
                        help="Explicit BlindSpot checkpoint path (overrides --resume)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=100,
                        help="Log per-step scalars and histograms every N steps")
    parser.add_argument("--vis-every", type=int, default=500,
                        help="Save visualization image to TensorBoard every N steps")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------
    base_dir = Path(args.root) / "training" / "blindspot"

    if args.run_name:
        run_name = args.run_name
    else:
        # Auto-number: run001, run002, ...
        existing = sorted(base_dir.glob("run[0-9][0-9][0-9]"))
        run_name = f"run{len(existing) + 1:03d}"

    run_dir = base_dir / run_name
    ckpt_dir = run_dir / "checkpoints"
    tb_dir = run_dir / "tensorboard"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    ckpt_last = str(ckpt_dir / "BlindSpot_last.pth")
    ckpt_best = str(ckpt_dir / "BlindSpot_best.pth")

    print(f"Run         : {run_dir}")
    print(f"Checkpoints : {ckpt_dir}")
    print(f"TensorBoard : {tb_dir}")
    print(f"  tensorboard --logdir {tb_dir}")

    # ------------------------------------------------------------------
    # Dataset + DataLoaders
    # ------------------------------------------------------------------

    metadata_dir = Path(args.metadata_dir)

    train_dataset = BlindSpotDataset(metadata_dir / "train_metadata.json")
    val_dataset = BlindSpotDataset(metadata_dir / "val_metadata.json")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
    )
    
    if len(train_loader) == 0:
        raise RuntimeError("Training DataLoader contains no batches.")

    if len(val_loader) == 0:
        raise RuntimeError("Validation DataLoader contains no batches.")

    steps_per_epoch = len(train_loader)

    print(f"Train samples      : {len(train_dataset):,}")
    print(f"Validation samples : {len(val_dataset):,}")
    print(f"Steps per epoch    : {steps_per_epoch:,}")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = BlindSpotTrainer(tensorboard_dir=str(tb_dir))
    trainer.zero_grad()

    # Resume state
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    
    resume_path = ""

    # An explicit --checkpoint takes precedence over --resume.
    if args.checkpoint:
        if Path(args.checkpoint).exists():
            resume_path = args.checkpoint
        else:
            print(f"  WARNING: --checkpoint not found: {args.checkpoint}")
    elif args.resume:
        if Path(ckpt_last).exists():
            resume_path = ckpt_last
        else:
            print("  --resume: no BlindSpot_last.pth found, starting fresh.")

    if resume_path:
        start_epoch, global_step, best_val_loss = trainer.load_checkpoint(resume_path)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{args.epochs}  "
              f"(global_step={global_step})")

        trainer.set_train_mode()
        trainer.reset_averages()

        p_bar = tqdm.tqdm(
            train_loader, total=steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        )

        for batch in p_bar:
            trainer.set_batch(batch)
            trainer.run_model()
            trainer.loss_backward()
            trainer.run_optimizer()

            global_step += 1

            p_bar.set_description(
                f"[{epoch + 1}/{args.epochs}]  "
                f"loss {trainer.avg_total.avg:.4f}  "
                f"|g| {trainer._grad_norm:.2f}"
            )

            if global_step % args.log_every == 0:
                trainer.log_train_step(global_step)
                trainer.log_histograms(global_step)

            if global_step % args.vis_every == 0:
                trainer.save_visualization(global_step)

        trainer.log_train_epoch(epoch + 1)

        # Save checkpoint every epoch
        trainer.save_checkpoint(ckpt_last, epoch + 1, global_step, best_val_loss)
        ckpt_epoch = str(ckpt_dir / f"BlindSpot_epoch{epoch + 1:03d}.pth")
        trainer.save_checkpoint(ckpt_epoch, epoch + 1, global_step, best_val_loss)

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        print("  Validating...")

        trainer.set_eval_mode()

        with torch.no_grad():
            val_loss, val_accuracy = _run_val(trainer, val_loader)

        trainer.log_val_epoch(val_loss, val_accuracy, epoch + 1)
        trainer.save_visualization(epoch + 1, split="val")

        print(
            f"  [Val] loss {val_loss:.4f}  "
            f"accuracy {val_accuracy:.2f}%"
        )

        # Track the best-performing checkpoint separately.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(ckpt_best, epoch + 1, global_step, best_val_loss)
            print(f"  *** New best val loss: {best_val_loss:.4f} → BlindSpot_best.pth")

        trainer.set_train_mode()

    trainer.cleanup()


if __name__ == "__main__":
    main()