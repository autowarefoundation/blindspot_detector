import json
import argparse
import logging
from pathlib import Path
from collections import Counter

from sklearn.model_selection import GroupShuffleSplit

RANDOM_STATE = 42
TRAIN_RATIO = 0.7
CLASS_BALANCE_THRESHOLD = 0.05  # 5%


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def build_samples(
    camera_dir: Path,
    camera_name: str,
):
    """
    Build temporal BlindSpot samples from a camera directory.

    Example
    -------
        00005_MVL.png
        00005_MVL_prev.png

    becomes

        {
            "timestep_id": "00005",
            "camera": "left",
            "image_prev": "...",
            "image_curr": "...",
            "occupancy": 1
        }
    """

    annotation_file = camera_dir / "occupancy_annotations.json"
    images_dir = camera_dir / "images"

    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotations not found: {annotation_file}")

    with annotation_file.open("r") as f:
        annotations = json.load(f)

    logger.info(f"Loaded {len(annotations)} annotations " f"for camera: {camera_name}")

    samples = []

    for key, occupancy in annotations.items():

        # Skip previous-frame entries.
        # The current-frame entry defines the temporal pair.
        if key.endswith("_prev"):
            continue

        timestep_id = key.split("_")[0]

        curr_image = images_dir / f"{key}.png"
        prev_image = images_dir / f"{key}_prev.png"

        if not curr_image.exists():
            logger.warning(f"Missing current image: {curr_image}")
            continue

        if not prev_image.exists():
            logger.warning(f"Missing previous image: {prev_image}")
            continue

        sample = {
            "timestep_id": timestep_id,
            "camera": camera_name,
            "image_prev": str(prev_image),
            "image_curr": str(curr_image),
            "occupancy": int(occupancy),
        }

        samples.append(sample)

    logger.info(f"Built {len(samples)} temporal samples " f"for camera: {camera_name}")

    return samples


def check_duplicate_samples(samples):
    """
    Detect duplicate temporal samples.

    A sample is uniquely identified by:

        timestep_id
        camera
        image_prev
        image_curr
    """

    logger.info("Checking for duplicate samples...")

    sample_keys = []

    for sample in samples:

        key = (
            sample["timestep_id"],
            sample["camera"],
            sample["image_prev"],
            sample["image_curr"],
        )

        sample_keys.append(key)

    unique_keys = set(sample_keys)

    duplicate_count = len(sample_keys) - len(unique_keys)

    if duplicate_count > 0:

        logger.error(f"Found {duplicate_count} duplicate samples.")

        raise RuntimeError("Duplicate samples detected in dataset.")

    logger.info("Duplicate sample check passed.")

    timestep_counts = Counter(sample["timestep_id"] for sample in samples)

    logger.info(f"Unique timesteps: {len(timestep_counts)}")

    logger.info(
        f"Average samples per timestep: " f"{len(samples) / len(timestep_counts):.2f}"
    )


def check_camera_coverage(
    left_samples,
    right_samples,
):
    """
    Verify timestep coverage between left and right cameras.

    Logs timesteps that exist only in one camera.
    """

    logger.info(
        "Checking left/right camera coverage..."
    )

    left_timesteps = {
        sample["timestep_id"]
        for sample in left_samples
    }

    right_timesteps = {
        sample["timestep_id"]
        for sample in right_samples
    }

    left_only = left_timesteps - right_timesteps
    right_only = right_timesteps - left_timesteps

    logger.info(
        f"Left camera timesteps : {len(left_timesteps)}"
    )

    logger.info(
        f"Right camera timesteps: {len(right_timesteps)}"
    )

    overlap = left_timesteps.intersection(
        right_timesteps
    )

    logger.info(
        f"Shared timesteps: {len(overlap)}"
    )

    if left_only:

        logger.warning(
            f"{len(left_only)} timesteps exist only "
            f"in the left camera."
        )

        logger.warning(
            f"Example left-only timesteps: "
            f"{sorted(list(left_only))[:10]}"
        )

    if right_only:

        logger.warning(
            f"{len(right_only)} timesteps exist only "
            f"in the right camera."
        )

        logger.warning(
            f"Example right-only timesteps: "
            f"{sorted(list(right_only))[:10]}"
        )

    if not left_only and not right_only:

        logger.info(
            "Camera coverage check passed."
        )


def split_dataset(samples):
    """
    Group-based train/validation split.

    Group key:
        timestep_id

    This guarantees that all variants sharing the same timestep:

        00005_MVL
        00005_MVL_prev
        00005_MVR
        00005_MVR_prev

    always remain in the same partition.
    """

    groups = [sample["timestep_id"] for sample in samples]

    unique_groups = len(set(groups))

    logger.info(f"Found {unique_groups} unique timestep groups")

    splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE,
    )

    train_idx, val_idx = next(
        splitter.split(
            samples,
            groups=groups,
        )
    )

    train_samples = [samples[i] for i in train_idx]

    val_samples = [samples[i] for i in val_idx]

    train_groups = {samples[i]["timestep_id"] for i in train_idx}

    val_groups = {samples[i]["timestep_id"] for i in val_idx}

    logger.info(f"Train groups: {len(train_groups)}")

    logger.info(f"Validation groups: {len(val_groups)}")

    # ------------------------------------------------------------------
    # Leakage check
    # ------------------------------------------------------------------
    overlap = train_groups.intersection(val_groups)

    if overlap:

        logger.error(
            f"Data leakage detected. "
            f"{len(overlap)} timestep groups appear in both "
            f"training and validation sets."
        )

        logger.error(f"Example overlapping groups: " f"{sorted(list(overlap))[:10]}")

        raise RuntimeError(
            "Group-based split failed. "
            "Train and validation sets share timestep groups."
        )

    logger.info(
        "Leakage check passed. "
        "No timestep groups are shared between train and validation."
    )

    return train_samples, val_samples


def check_class_distribution(
    train_samples,
    val_samples,
    threshold=CLASS_BALANCE_THRESHOLD,
):
    """
    Compare occupancy distribution between train and validation.

    Warn if the occupancy ratio differs by more than the specified
    threshold.

    Example:
        threshold = 0.05 -> warn if difference exceeds 5%
    """

    train_occ_ratio = sum(s["occupancy"] for s in train_samples) / len(train_samples)

    val_occ_ratio = sum(s["occupancy"] for s in val_samples) / len(val_samples)

    diff = abs(train_occ_ratio - val_occ_ratio)

    logger.info(
        f"Occupancy ratio: "
        f"train={train_occ_ratio:.1%}, "
        f"val={val_occ_ratio:.1%}, "
        f"diff={diff:.1%}"
    )

    if diff > threshold:

        logger.warning(
            f"Class distribution imbalance detected. "
            f"Train and validation occupancy ratios differ "
            f"by {diff:.1%} (> {threshold:.1%})."
        )

    else:

        logger.info("Class distribution check passed.")


def print_dataset_statistics(
    name,
    samples,
):
    total = len(samples)

    occupied = sum(sample["occupancy"] for sample in samples)

    free = total - occupied

    occupancy_ratio = occupied / total if total > 0 else 0.0

    logger.info(
        f"{name}: "
        f"samples={total}, "
        f"occupied={occupied} ({occupancy_ratio:.1%}), "
        f"free={free}"
    )


def convert(
    input_ds_dir,
    output_ds_dir,
):
    """
    Parse BlindSpot dataset and generate train/validation metadata.
    """

    input_ds_dir = Path(input_ds_dir)
    output_ds_dir = Path(output_ds_dir)

    left_dir = input_ds_dir / "left_fisheye"
    right_dir = input_ds_dir / "right_fisheye"

    logger.info("Parsing left camera...")
    left_samples = build_samples(
        left_dir,
        camera_name="left",
    )

    logger.info("Parsing right camera...")
    right_samples = build_samples(
        right_dir,
        camera_name="right",
    )

    # Verify left/right camera coverage
    check_camera_coverage(
        left_samples,
        right_samples,
    )

    # Combine left and right cameras into a single dataset
    samples = left_samples + right_samples

    logger.info(f"Total parsed samples: {len(samples)}")

    if len(samples) == 0:
        raise RuntimeError("No valid BlindSpot samples were found.")

    # Check for duplicated samples
    check_duplicate_samples(samples)

    # Group-based split
    train_samples, val_samples = split_dataset(samples)

    # Verify train/validation class balance
    check_class_distribution(
        train_samples,
        val_samples,
    )

    logger.info(f"Train samples: {len(train_samples)}")

    logger.info(f"Validation samples: {len(val_samples)}")

    output_ds_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_file = output_ds_dir / "train_metadata.json"
    val_file = output_ds_dir / "val_metadata.json"

    with train_file.open("w") as f:
        json.dump(
            train_samples,
            f,
            indent=4,
        )

    with val_file.open("w") as f:
        json.dump(
            val_samples,
            f,
            indent=4,
        )

    print_dataset_statistics(
        "Training Set",
        train_samples,
    )

    print_dataset_statistics(
        "Validation Set",
        val_samples,
    )

    logger.info(f"Saved training metadata: {train_file}")

    logger.info(f"Saved validation metadata: {val_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_ds_dir",
        required=True,
        help="Path to WoodScape/data/all_images",
    )

    parser.add_argument(
        "-o",
        "--output_ds_dir",
        required=True,
        help="Output directory for metadata files",
    )

    args = parser.parse_args()

    convert(
        args.input_ds_dir,
        args.output_ds_dir,
    )
