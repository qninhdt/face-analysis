import torch


class YOLO(torch.nn.Module):
    def __init__(self, backbone=None, neck=None, head=None, loss=None):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.loss = loss

    def forward(self, x, targets=None):
        x = self.backbone(x)

        if self.neck is not None:
            x = self.neck(x)

        x = self.head(x)

        x = self.loss(x, targets)

        return x
