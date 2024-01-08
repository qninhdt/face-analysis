from typing import List, Literal, Dict
from pathlib import Path

import json

import numpy as np
import torch
from torchvision.transforms import v2 as T
from torchvision import tv_tensors
from torch.utils.data import Dataset
from PIL.Image import Image, open as open_image


class PIXTAFaceDataset(Dataset):
    def __init__(self, data_dir: str, type: Literal["train", "test", "val", "private"]) -> None:
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.type = type
    
        with open(self.data_dir / "labels.json") as f:
            labels = json.load(f)
            images = {}

            for image in labels["images"]:
                images[image["id"]] = image

            self.targets: List[dict] = [images[id] for id in labels[type]]
            self.categories: Dict[list] = labels["categories"]

    def __getitem__(self, idx: int) -> dict:
        target = self.targets[idx]

        image_path = self.data_dir / 'images' / target['file_name']

        # load image
        orig_image: Image = open_image(image_path).convert('RGB')
        image = torch.tensor(np.array(orig_image, dtype=np.uint8)).permute(2, 0, 1)

        w, h = orig_image.size

        age = torch.tensor([face['age'] for face in target['faces']], dtype=torch.int32)
        gender = torch.tensor([face['gender'] for face in target['faces']], dtype=torch.int32)
        race = torch.tensor([face['race'] for face in target['faces']], dtype=torch.int32)
        emotion = torch.tensor([face['emotion'] for face in target['faces']], dtype=torch.int32)
        masked = torch.tensor([face['masked'] for face in target['faces']], dtype=torch.int32)
        skintone = torch.tensor([face['skintone'] for face in target['faces']], dtype=torch.int32)
        boxes = tv_tensors.BoundingBoxes([face['box'] for face in target['faces']], format='xywh',
                                         canvas_size=(h, w))
        area = torch.tensor([face['box'][2] * face['box'][3] for face in target['faces']], dtype=torch.float32)

        sample = {
            'image': image,
            'age': age,
            'gender': gender,
            'race': race,
            'emotion': emotion,
            'masked': masked,
            'skintone': skintone,
            'boxes': boxes,
            'area': area,
        }

        return sample

    def __len__(self) -> int:
        # if self.type == "train":
        #     return 10
        # if self.type == "val":
        #     return 1000
        return len(self.targets)
