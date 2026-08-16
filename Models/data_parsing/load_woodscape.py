import json
import logging
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 512

CROP_TOP = 70
CROP_BOTTOM = 256


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


class BlindSpotDataset(Dataset):
    """
    BlindSpot occupancy dataset.

    Each sample consists of:

        image_prev
        image_curr
        occupancy

    where occupancy:

        0 -> FREE
        1 -> OCCUPIED
    """

    def __init__(
        self,
        metadata_file,
        transform=None,
    ):
        super().__init__()

        self.metadata_file = Path(metadata_file)

        if not self.metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_file}"
            )

        with self.metadata_file.open("r") as f:
            self.samples = json.load(f)

        logger.info(
            f"Loaded {len(self.samples)} samples "
            f"from {self.metadata_file}"
        )

        occupied = sum(
            sample["occupancy"]
            for sample in self.samples
        )

        free = len(self.samples) - occupied

        logger.info(
            f"Dataset statistics: "
            f"FREE={free}, OCCUPIED={occupied}"
        )

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def crop_and_resize(image):
        """
        Original:
            1280 x 966

        Crop:
            Top    : 70 px
            Bottom : 256 px

        Result:
            1280 x 640

        Resize:
            1024 x 512
        """

        width, height = image.size

        image = image.crop(
            (
                0,                  # left
                CROP_TOP,           # top
                width,              # right
                height - CROP_BOTTOM
            )
        )

        image = image.resize(
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            Image.Resampling.LANCZOS,
        )

        return image

    def load_image(
        self,
        image_path,
    ):
        image = Image.open(image_path).convert("RGB")

        image = self.crop_and_resize(image)

        image = self.transform(image)

        return image

    def __getitem__(
        self,
        idx,
    ):
        sample = self.samples[idx]

        image_prev = self.load_image(
            sample["image_prev"]
        )

        image_curr = self.load_image(
            sample["image_curr"]
        )

        occupancy = torch.tensor(
            sample["occupancy"],
            dtype=torch.long,
        )

        return {
            "image_prev": image_prev,
            "image_curr": image_curr,
            "occupancy": occupancy,
        }


def create_dataloader(
    metadata_file,
    batch_size=32,
    shuffle=False,
    num_workers=4,
):
    """
    Convenience function for creating a DataLoader.
    """

    dataset = BlindSpotDataset(
        metadata_file=metadata_file,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


if __name__ == "__main__":

    train_loader = create_dataloader(
        metadata_file="train_metadata.json",
        batch_size=8,
        shuffle=True,
    )

    batch = next(iter(train_loader))

    print(
        "image_prev:",
        batch["image_prev"].shape,
    )

    print(
        "image_curr:",
        batch["image_curr"].shape,
    )

    print(
        "occupancy:",
        batch["occupancy"].shape,
    )