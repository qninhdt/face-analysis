import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class AveragePrecision(MeanAveragePrecision):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _insert_dummy_labels(self, targets):
        new_targets = []
        for target in targets:
            dummy_labels = torch.zeros(
                (target["boxes"].shape[0]),
                dtype=torch.int64,
                device=target["boxes"].device,
            )

            target = {**target, "labels": dummy_labels}
            new_targets.append(target)

        return new_targets

    def update(self, preds, targets) -> None:
        for target in targets:
            target["boxes"] = torch.clone(target["boxes"]).detach()
        new_targets = self._insert_dummy_labels(targets)
        preds = self._insert_dummy_labels(preds)

        super().update(preds, new_targets)
