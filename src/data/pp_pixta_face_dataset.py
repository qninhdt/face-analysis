import json
from pathlib import Path

import torch
import numpy as np
from PIL.Image import Image, open as open_image
from torch.utils.data import Dataset    
from torchvision.transforms import v2 as T

from utils.transform import SquarePad

IMAGE_SIZE = 640

class PPPIXTAFaceDataset(Dataset):

    def __init__(self, data_dir: str) -> None:
        super().__init__()

        self.data_dir = Path(data_dir)

        with open(self.data_dir / "file_name_to_image_id.json") as f:
            self.image_map = json.load(f)
        
        # image list sort by image_id
        self.image_list = list(sorted(self.image_map.items(), key=lambda x: x[1]))

        self.transforms = T.Compose([
            SquarePad(),
            T.Resize(IMAGE_SIZE, antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.6596, 0.6235, 0.5875], std=[0.2272, 0.2248, 0.2345])
        ])

    def __getitem__(self, index: int) -> torch.Tensor:
        file_name, image_id = self.image_list[index]

        image_path = self.data_dir / 'images' / file_name
        # load image
        orig_image = open_image(image_path).convert('RGB')
        image = torch.tensor(np.array(orig_image, dtype=np.uint8)).permute(2, 0, 1)

        w, h = orig_image.size

        image = self.transforms(image)

        label = {
            "file_name": file_name,
            "image_id": image_id,
            "width": w,
            "height": h,
        }

        return image, label

    def __len__(self) -> int:
        return len(self.image_list)

    