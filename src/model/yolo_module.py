from typing import Any, Dict, Optional, Tuple, List

import time
import torch
from lightning import LightningModule
# Train
from utils.ema import ModelEMA
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
# Evaluate
from utils.flops import model_summary
from utils.metrics import AveragePrecision
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
        
        # self.automatic_optimization = False

        # metrics
        iou_types = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        self.val_AP = AveragePrecision('cxcywh', 'bbox', iou_types)
        self.test_AP = AveragePrecision('cxcywh', 'bbox', iou_types)

        # Test
        # if test_cfgs is not None:
        #     self.visualize = test_cfgs['visualize']
        #     self.test_nms = test_cfgs['test_nms']
        #     self.test_conf = test_cfgs['test_conf']
        #     self.show_dir = test_cfgs['show_dir']
        #     self.show_score_thr = test_cfgs['show_score_thr']

    def on_train_start(self) -> None:
        # if self.hparams.optimizer['ema'] is True:
        #     self.ema_model = ModelEMA(self.model, 0.9998)

        # model_summary(self, self.img_size, self.device)
        pass

    def training_step(self, batch, batch_idx):
        images, targets = batch

        losses = self.model(images, targets)

        # self.log_dict(losses)
        self.log("train/loss", losses['loss'], prog_bar=True)
        self.log("train/box_loss", losses['box_loss'], prog_bar=True)
        self.log("train/age_loss", losses['age_loss'], prog_bar=True)
        self.log("train/race_loss", losses['race_loss'], prog_bar=True)
        self.log("train/masked_loss", losses['masked_loss'], prog_bar=True)
        self.log("train/skintone_loss", losses['skintone_loss'], prog_bar=True)
        self.log("train/emotion_loss", losses['emotion_loss'], prog_bar=True)
        self.log("train/gender_loss", losses['gender_loss'], prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)

        return losses['loss']

    def on_validation_start(self) -> None:
        self.val_AP.reset()

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # inference
        start_time = time.time()
        predictions = self.model(images, targets)
        self.infr_times.append(time.time() - start_time)

        # postprocess
        start_time = time.time()
        predictions = postprocess(predictions, self.confidence_threshold, self.nms_threshold)
        self.nms_times.append(time.time() - start_time)

        self.val_AP(predictions, targets)

        if batch_idx == self.trainer.num_val_batches[0] - 1:
            metrics = self.val_AP.compute()
            metrics = {k: v.to(self.device) for k, v in metrics.items()}

            self.log("val/AP", metrics['map'], prog_bar=True)



    def on_validation_end(self) -> None:
        average_ifer_time = torch.tensor(self.infr_times, dtype=torch.float32).mean().item()
        average_nms_time = torch.tensor(self.nms_times, dtype=torch.float32).mean().item()
        print("The average iference time is %.4fs, nms time is %.4fs" % (average_ifer_time, average_nms_time))
        self.infr_times, self.nms_times = [], []

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),
                        lr=self.hparams.optimizer['learning_rate'])
        
        # total_steps = self.trainer.estimated_stepping_batches
        
        lr_scheduler = ExponentialLR(optimizer, gamma=self.hparams.optimizer['gamma'])
        
        return [optimizer], [lr_scheduler]

    def on_train_end(self) -> None:
        pass

    def forward(self, imgs):
        self.model.eval()
        detections = self.model(imgs)
        return detections
    
    def on_test_start(self) -> None:
        self.test_AP.reset()

    def test_step(self, batch, batch_idx):
        images, targets = batch

        # inference
        start_time = time.time()
        predictions = self.model(images, targets)
        self.infr_times.append(time.time() - start_time)

        # postprocess
        start_time = time.time()
        predictions = postprocess(predictions, self.confidence_threshold, self.nms_threshold)
        self.nms_times.append(time.time() - start_time)

        self.test_AP(predictions, targets)

        match_idx = match_pred_boxes(predictions, targets)
        print("match_idx", match_idx)

        if batch_idx == self.trainer.num_test_batches[0] - 1:
            metrics = self.test_AP.compute()
            metrics = {k: v.to(self.device) for k, v in metrics.items()}

            self.log("test/AP", metrics['map'], prog_bar=True)

            

    def on_test_epoch_end(self):
        average_ifer_time = torch.tensor(self.infr_times, dtype=torch.float32).mean().item()
        average_nms_time = torch.tensor(self.nms_times, dtype=torch.float32).mean().item()
        print("The average iference time is %.4fs, nms time is %.4fs" % (average_ifer_time, average_nms_time))
        self.infr_times, self.nms_times = [], []
