from typing import Any, Dict, Optional, Tuple, List

import time
import torch
from lightning import LightningModule

# Train
from utils.ema import ModelEMA
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, ConstantLR, ReduceLROnPlateau
from model.layers.lr_scheduler import CosineWarmupScheduler

# Evaluate
from utils.flops import model_summary
from utils.metrics import AveragePrecision
from torchmetrics.classification import Accuracy
from torchmetrics import MeanMetric
from model.evaluators.postprocess import postprocess, match_pred_boxes


class YOLOModule(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        image_size: int,
        nms_threshold: float,
        confidence_threshold: float,
        optimizer: Dict[str, Any],
        compile: bool = True,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.model = model
        self.img_size = image_size
        self.nms_threshold = nms_threshold
        self.confidence_threshold = confidence_threshold
        self.infr_times = []
        self.nms_times = []
        self.ema_model = None

        self.automatic_optimization = False

        # metrics
        iou_types = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

        self.val_AP = AveragePrecision("cxcywh", "bbox", iou_types)

        self.val_age_acc = Accuracy(task="multiclass", num_classes=6)
        self.val_race_acc = Accuracy(task="multiclass", num_classes=3)
        self.val_masked_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_skintone_acc = Accuracy(task="multiclass", num_classes=4)
        self.val_emotion_acc = Accuracy(task="multiclass", num_classes=7)
        self.val_gender_acc = Accuracy(task="multiclass", num_classes=2)

        self.test_AP = AveragePrecision("cxcywh", "bbox", iou_types)
        self.test_age_acc = Accuracy(task="multiclass", num_classes=6)
        self.test_race_acc = Accuracy(task="multiclass", num_classes=3)
        self.test_masked_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_skintone_acc = Accuracy(task="multiclass", num_classes=4)
        self.test_emotion_acc = Accuracy(task="multiclass", num_classes=7)
        self.test_gender_acc = Accuracy(task="multiclass", num_classes=2)

        self.mean_loss = MeanMetric()
        self.mean_box_loss = MeanMetric()
        self.mean_obj_loss = MeanMetric()
        self.mean_age_loss = MeanMetric()
        self.mean_race_loss = MeanMetric()
        self.mean_masked_loss = MeanMetric()
        self.mean_skintone_loss = MeanMetric()
        self.mean_emotion_loss = MeanMetric()
        self.mean_gender_loss = MeanMetric()

        # Test
        # if test_cfgs is not None:
        #     self.visualize = test_cfgs['visualize']
        #     self.test_nms = test_cfgs['test_nms']
        #     self.test_conf = test_cfgs['test_conf']
        #     self.show_dir = test_cfgs['show_dir']
        #     self.show_score_thr = test_cfgs['show_score_thr']

    def on_train_start(self) -> None:
        if self.hparams.optimizer["ema"] is True:
            self.ema_model = ModelEMA(self.model, 0.9998)

        # model_summary(self, self.img_size, self.device)
        pass

    def training_step(self, batch, batch_idx):
        images, targets = batch

        losses = self.model(images, targets)

        self.log("train/rt_loss", losses["loss"], prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)

        self.mean_loss(losses["loss"])
        self.mean_box_loss(losses["box_loss"])
        self.mean_obj_loss(losses["obj_loss"])
        self.mean_age_loss(losses["age_loss"])
        self.mean_race_loss(losses["race_loss"])
        self.mean_masked_loss(losses["masked_loss"])
        self.mean_skintone_loss(losses["skintone_loss"])
        self.mean_emotion_loss(losses["emotion_loss"])
        self.mean_gender_loss(losses["gender_loss"])

        # Backward
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(losses["loss"])
        optimizer.step()

        if self.hparams.optimizer["ema"] is True:
            self.ema_model.update(self.model)
        self.lr_schedulers().step()

        if batch_idx == self.trainer.num_training_batches - 1:
            self.log(
                "train/loss", self.mean_loss.compute(), prog_bar=True, sync_dist=True
            )
            self.log(
                "train/box_loss",
                self.mean_box_loss.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "train/obj_loss",
                self.mean_obj_loss.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "train/age_loss",
                self.mean_age_loss.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "train/race_loss",
                self.mean_race_loss.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "train/masked_loss",
                self.mean_masked_loss.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "train/skintone_loss",
                self.mean_skintone_loss.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "train/emotion_loss",
                self.mean_emotion_loss.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "train/gender_loss",
                self.mean_gender_loss.compute(),
                prog_bar=True,
                sync_dist=True,
            )

        # return losses["loss"]

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # inference
        start_time = time.time()
        predictions = self.model(images)
        self.infr_times.append(time.time() - start_time)

        # postprocess
        start_time = time.time()
        predictions = postprocess(
            predictions, self.confidence_threshold, self.nms_threshold
        )
        self.nms_times.append(time.time() - start_time)

        self.val_AP.update(predictions, targets)

        match_idx = match_pred_boxes(predictions, targets)

        for i in range(len(predictions)):
            if match_idx[i].shape[0] == 0:
                continue
            self.val_age_acc(
                predictions[i]["age"][match_idx[i][:, 0]],
                targets[i]["age"][match_idx[i][:, 1]],
            )
            self.val_race_acc(
                predictions[i]["race"][match_idx[i][:, 0]],
                targets[i]["race"][match_idx[i][:, 1]],
            )
            self.val_masked_acc(
                predictions[i]["masked"][match_idx[i][:, 0]],
                targets[i]["masked"][match_idx[i][:, 1]],
            )
            self.val_skintone_acc(
                predictions[i]["skintone"][match_idx[i][:, 0]],
                targets[i]["skintone"][match_idx[i][:, 1]],
            )
            self.val_emotion_acc(
                predictions[i]["emotion"][match_idx[i][:, 0]],
                targets[i]["emotion"][match_idx[i][:, 1]],
            )
            self.val_gender_acc(
                predictions[i]["gender"][match_idx[i][:, 0]],
                targets[i]["gender"][match_idx[i][:, 1]],
            )

        if batch_idx == self.trainer.num_val_batches[0] - 1:
            metrics = self.val_AP.compute()
            metrics = {k: v.to(self.device) for k, v in metrics.items()}

            self.log("val/AP", metrics["map"], prog_bar=True, sync_dist=True)
            self.log("val/AP_50", metrics["map_50"], prog_bar=True, sync_dist=True)
            self.log(
                "val/age_acc", self.val_age_acc.compute(), prog_bar=True, sync_dist=True
            )
            self.log(
                "val/race_acc",
                self.val_race_acc.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "val/masked_acc",
                self.val_masked_acc.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "val/skintone_acc",
                self.val_skintone_acc.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "val/emotion_acc",
                self.val_emotion_acc.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "val/gender_acc",
                self.val_gender_acc.compute(),
                prog_bar=True,
                sync_dist=True,
            )

    def on_validation_end(self) -> None:
        average_ifer_time = (
            torch.tensor(self.infr_times, dtype=torch.float32).mean().item()
        )
        average_nms_time = (
            torch.tensor(self.nms_times, dtype=torch.float32).mean().item()
        )
        print(
            "The average iference time is %.4fs, nms time is %.4fs"
            % (average_ifer_time, average_nms_time)
        )
        self.infr_times, self.nms_times = [], []

    def configure_optimizers(self):
        # optimizer = Adam(
        #     self.parameters(),
        #     lr=self.hparams.optimizer["learning_rate"],
        #     weight_decay=self.hparams.optimizer["weight_decay"],
        # )

        # scheduler = ReduceLROnPlateau(
        #     optimizer, mode="max", factor=0.5, patience=15, min_lr=0.00001, cooldown=5
        # )

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "val/AP",
        #         "interval": "epoch",
        #         "frequency": 1,
        #     },
        # }
        optimizer = SGD(
            self.parameters(),
            lr=self.hparams.optimizer["learning_rate"],
            momentum=self.hparams.optimizer["momentum"],
            weight_decay=self.hparams.optimizer["weight_decay"],
        )
        total_steps = self.trainer.estimated_stepping_batches
        lr_scheduler = CosineWarmupScheduler(
            optimizer,
            warmup=self.hparams.optimizer["warmup"] * total_steps,
            max_iters=total_steps,
        )
        return [optimizer], [lr_scheduler]

    def on_train_end(self) -> None:
        pass

    def forward(self, x):
        imgs, labels = x if isinstance(x, tuple) else (x, None)
        self.model.eval()
        detections = self.model(imgs)
        return detections, labels

    def test_step(self, batch, batch_idx):
        images, targets = batch

        # inference
        start_time = time.time()
        predictions = self.model(images)
        self.infr_times.append(time.time() - start_time)

        # postprocess
        start_time = time.time()
        predictions = postprocess(
            predictions, self.confidence_threshold, self.nms_threshold
        )
        self.nms_times.append(time.time() - start_time)

        self.test_AP(predictions, targets)

        match_idx = match_pred_boxes(predictions, targets)

        for i in range(len(predictions)):
            if match_idx[i].shape[0] == 0:
                continue
            self.test_age_acc(
                predictions[i]["age"][match_idx[i][:, 0]],
                targets[i]["age"][match_idx[i][:, 1]],
            )
            self.test_race_acc(
                predictions[i]["race"][match_idx[i][:, 0]],
                targets[i]["race"][match_idx[i][:, 1]],
            )
            self.test_masked_acc(
                predictions[i]["masked"][match_idx[i][:, 0]],
                targets[i]["masked"][match_idx[i][:, 1]],
            )
            self.test_skintone_acc(
                predictions[i]["skintone"][match_idx[i][:, 0]],
                targets[i]["skintone"][match_idx[i][:, 1]],
            )
            self.test_emotion_acc(
                predictions[i]["emotion"][match_idx[i][:, 0]],
                targets[i]["emotion"][match_idx[i][:, 1]],
            )
            self.test_gender_acc(
                predictions[i]["gender"][match_idx[i][:, 0]],
                targets[i]["gender"][match_idx[i][:, 1]],
            )

        if batch_idx == self.trainer.num_test_batches[0] - 1:
            metrics = self.test_AP.compute()
            metrics = {k: v.to(self.device) for k, v in metrics.items()}

            self.log("test/AP", metrics["map"], prog_bar=True, sync_dist=True)
            self.log("test/AP_50", metrics["map_50"], prog_bar=True, sync_dist=True)
            self.log(
                "test/age_acc",
                self.test_age_acc.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "test/race_acc",
                self.test_race_acc.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "test/masked_acc",
                self.test_masked_acc.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "test/skintone_acc",
                self.test_skintone_acc.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "test/emotion_acc",
                self.test_emotion_acc.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "test/gender_acc",
                self.test_gender_acc.compute(),
                prog_bar=True,
                sync_dist=True,
            )

    def on_test_epoch_end(self):
        average_ifer_time = (
            torch.tensor(self.infr_times, dtype=torch.float32).mean().item()
        )
        average_nms_time = (
            torch.tensor(self.nms_times, dtype=torch.float32).mean().item()
        )
        print(
            "The average iference time is %.4fs, nms time is %.4fs"
            % (average_ifer_time, average_nms_time)
        )
        self.infr_times, self.nms_times = [], []
