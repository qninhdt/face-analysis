from typing import Any, Dict

import torchvision.transforms.v2 as T
import torch
import torch.nn as nn
from torchvision.tv_tensors import BoundingBoxes
from torchvision.transforms import functional as F
from numpy import random

class SquarePad(nn.Module):
    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if type(sample) != torch.Tensor:
            image = sample['image']
        else:
            image = sample

        h = image.shape[1]
        w = image.shape[2]

        if h > w:
            r = (h - w) // 2
            l = (h - w) - r
            pad = T.Pad((l, 0, r, 0))
        else:
            t = (w - h) // 2
            b = (w - h) - t
            pad = T.Pad((0, t, 0, b))

        return T.Compose([pad])(sample)
    
class RandomCropWithoutLossingBoxes(nn.Module):

    def get_maximum_box(self, boxes: BoundingBoxes) -> torch.Tensor:
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2] + boxes[:, 0], boxes[:, 3] + boxes[:, 1]
        return torch.stack([x1.min(), y1.min(), x2.max(), y2.max()])

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        maximum_box = self.get_maximum_box(sample['boxes'])
    
        h, w = sample['image'].shape[1:3]

        x1 = random.randint(0, maximum_box[0] + 1)
        y1 = random.randint(0, maximum_box[1] + 1)
        x2 = random.randint(maximum_box[2], w + 1)
        y2 = random.randint(maximum_box[3], h + 1)

        sample['image'] = F.crop(sample['image'], y1, x1, y2 - y1, x2 - x1)
        sample['boxes'][:, 0] -= x1
        sample['boxes'][:, 1] -= y1
        sample['boxes'].canvas_size = (y2 - y1, x2 - x1)

        return sample
    
class Normalize(nn.Module):
    def __init__(self, mean: list, std: list) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        boxes = sample['boxes']

        # convert XYWH to CXCYWH        
        boxes = boxes.to(torch.float32)
        boxes[:, 0] += boxes[:, 2] / 2
        boxes[:, 1] += boxes[:, 3] / 2
        boxes.format = 'cxcywh'

        sample['boxes'] = boxes

        return T.Compose([T.Normalize(self.mean, self.std)])(sample)
