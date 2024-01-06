from typing import List, Tuple, Callable
from pathlib import Path

import json

import torch
import numpy as np
from random import Random
from torchvision.transforms import v2 as T
from torchvision import tv_tensors
from torch.utils.data import Dataset
from PIL.Image import Image, open as open_image


class UTKFaceDataset(Dataset):
    def __init__(self, data_dir: str) -> None:
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.image_paths = self.load_dataset()

        for _ in range(4):
            Random(1).shuffle(self.image_paths)

        print(self.data_dir)

    def load_dataset(self) -> List[str]:
        files = self.data_dir.glob("*.jpg")
        return [file.name for file in files]
    
    def __getitem__(self, idx: int) -> dict:
        path = self.image_paths[idx]
        labels = path.split("_")

        # load image
        image: Image = open_image(
            self.data_dir / path
        ).convert("RGB")

        # convert PIL image to torch tensor
        image = torch.tensor(np.array(image, dtype=np.uint8)).permute(2, 0, 1)

        age = torch.tensor(int(labels[0]), dtype=torch.float32)
        gender = torch.tensor(int(labels[1]), dtype=torch.float32)
        race = torch.tensor(int(labels[2]), dtype=torch.float32)

        sample = {
            'image': image,
            'age': age,
            'gender': gender,
            'race': race
        }

        return sample

    def __len__(self) -> int:
        return len(self.image_paths)
