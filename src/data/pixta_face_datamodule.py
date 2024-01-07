from typing import Any, Dict, Optional, Tuple, List

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2 as T
from utils.dataset import ApplyTransform
from utils.transform import SquarePad, Normalize, RandomCropWithoutLossingBoxes

from .pixta_face_dataset import PIXTAFaceDataset

IMAGE_SIZE = 640


class PIXTAFaceDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.save_hyperparameters(logger=False)

        # data transformations
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.train_transforms = T.Compose([
            # RandomCropWithoutLossingBoxes(),
            SquarePad(),
            T.RandomRotation(degrees=[-30, 30]),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomGrayscale(p=0.1),
            T.Resize(IMAGE_SIZE, antialias=True),
            T.ColorJitter(brightness=[0.5, 1.25]),
            # T.RandomApply([
            #     T.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2))
            # ], p=0.5),
            T.ToDtype(torch.float32, scale=True),
            normalize,
        ])

        self.transforms = T.Compose([
            SquarePad(),
            T.Resize(IMAGE_SIZE, antialias=True),
            T.ToDtype(torch.float32, scale=True),
            normalize,
        ])

        self.dataset: Optional[PIXTAFaceDataset] = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes()

    def prepare_data(self) -> None:
        return

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.batch_size // self.trainer.world_size
            )

        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = PIXTAFaceDataset(self.data_dir, "train")
            self.data_val = PIXTAFaceDataset(self.data_dir, "val")
            self.data_test = PIXTAFaceDataset(self.data_dir, "test")

            self.data_train = ApplyTransform(self.data_train, self.train_transforms)
            self.data_val = ApplyTransform(self.data_val, self.transforms)
            self.data_test = ApplyTransform(self.data_test, self.transforms)

    def train_dataloader(self) -> DataLoader[Any]:
        return self._create_dataloader(self.data_train, self.batch_size_per_device)

    def val_dataloader(self) -> DataLoader[Any]:
        return self._create_dataloader(self.data_val, self.batch_size_per_device)

    def test_dataloader(self) -> DataLoader[Any]:
        return self._create_dataloader(self.data_test, self.batch_size_per_device)

    def _create_dataloader(self, dataset: Dataset, batch_size: int) -> DataLoader[Any]:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, List[dict]]:
        images = torch.stack([x['image'] for x in batch])
        targets = [{k: v for k, v in x.items() if k != 'image'} for x in batch]

        return images, targets
