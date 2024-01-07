from typing import Any, Dict, Optional, Tuple, List

import time
import torch
from lightning import LightningModule
# Train
from utils.ema import ModelEMA
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR
# Evaluate
from utils.flops import model_summary
from model.evaluators.postprocess import postprocess, format_outputs

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

        self.save_hyperparameters(logger=False, ignore=['model'])

        self.model = model
        self.img_size = image_size
        self.nms_threshold = nms_threshold
        self.confidence_threshold = confidence_threshold
        self.infr_times = []
        self.nms_times = []
        self.ema_model = None
        self.ap50_95 = 0
        self.ap50 = 0
        # self.automatic_optimization = False

        # Test
        # if test_cfgs is not None:
        #     self.visualize = test_cfgs['visualize']
        #     self.test_nms = test_cfgs['test_nms']
        #     self.test_conf = test_cfgs['test_conf']
        #     self.show_dir = test_cfgs['show_dir']
        #     self.show_score_thr = test_cfgs['show_score_thr']

    def on_train_start(self) -> None:
        if self.hparams.optimizer['ema'] is True:
            self.ema_model = ModelEMA(self.model, 0.9998)

        model_summary(self.model, self.img_size, self.device)

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

        # backward
        # optimizer = self.optimizers()
        # optimizer.zero_grad()

        # self.manual_backward(losses['loss'])
        # optimizer.step()

        # if self.hparams.optimizer['ema'] is True:
        #     self.ema_model.update(self.model)

        # self.lr_schedulers().step()

        return losses['loss']

    def validation_step(self, batch, batch_idx):
    #     imgs, labels, img_hw, image_id, img_name = batch
    #     if self.ema_model is not None:
    #         model = self.ema_model.ema
    #     else:
    #         model = self.model
    #     start_time = time.time()
    #     detections = model(imgs, labels)
    #     self.infr_times.append(time.time() - start_time)
    #     start_time = time.time()
    #     detections = postprocess(detections, self.confidence_threshold, self.nms_threshold)
    #     self.nms_times.append(time.time() - start_time)
    #     json_det, det = format_outputs(detections, image_id, img_hw, self.img_size_val,
    #                                    self.trainer.datamodule.dataset_val.class_ids, labels)
    #     return json_det, det
        return torch.nn.Parameter(torch.tensor(0.0))

    # def validation_epoch_end(self, val_step_outputs):
    #     json_list = []
    #     det_list = []
    #     for i in range(len(val_step_outputs)):
    #         json_list += val_step_outputs[i][0]
    #         det_list += val_step_outputs[i][1]
    #     # COCO Evaluator
    #     ap50_95, ap50, summary = COCOEvaluator(json_list, self.trainer.datamodule.dataset_val)
    #     print("Batch {:d}, mAP = {:.3f}, mAP50 = {:.3f}".format(self.current_epoch, ap50_95, ap50))
    #     print(summary)
    #     # VOC Evaluator
    #     VOCEvaluator(det_list, self.trainer.datamodule.dataset_val, iou_thr=0.5)

    #     self.log("mAP", ap50_95, prog_bar=False)
    #     self.log("mAP50", ap50, prog_bar=False)
    #     if ap50_95 > self.ap50_95:
    #         self.ap50_95 = ap50_95
    #     if ap50 > self.ap50:
    #         self.ap50 = ap50

    #     average_ifer_time = torch.tensor(self.infr_times, dtype=torch.float32).mean().item()
    #     average_nms_time = torch.tensor(self.nms_times, dtype=torch.float32).mean().item()
    #     print("The average iference time is %.4fs, nms time is %.4fs" % (average_ifer_time, average_nms_time))
    #     self.infr_times, self.nms_times = [], []

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),
                        lr=self.hparams.optimizer['learning_rate'])
        
        total_steps = self.trainer.estimated_stepping_batches
        
        lr_scheduler = ExponentialLR(optimizer, gamma=self.hparams.optimizer['gamma'])
        
        return [optimizer], [lr_scheduler]

    def on_train_end(self) -> None:
        print("Best mAP = {:.3f}, best mAP50 = {:.3f}".format(self.ap50_95, self.ap50))

    def forward(self, imgs):
        self.model.eval()
        detections = self.model(imgs)
        return detections

    def test_step(self, batch, batch_idx):
    #     imgs, labels, img_hw, image_id, img_name = batch
    #     # inference
    #     model = self.model
    #     start_time = time.time()
    #     detections = model(imgs, labels)
    #     self.infr_times.append(time.time() - start_time)
    #     start_time = time.time()
    #     # postprocess
    #     detections = postprocess(detections, self.test_conf, self.test_nms)
    #     self.nms_times.append(time.time() - start_time)
    #     json_det, det = format_outputs(detections, image_id, img_hw, self.img_size_val,
    #                                    self.trainer.datamodule.dataset_test.class_ids, labels)
    #     return json_det, det, imgs, img_name
        return torch.nn.Parameter(torch.tensor(0.0))

    # def test_epoch_end(self, test_step_outputs):
    #     json_list = []
    #     det_list = []
    #     for i in range(len(test_step_outputs)):
    #         json_list += test_step_outputs[i][0]
    #         det_list += test_step_outputs[i][1]
    #     # COCO Evaluator
    #     ap50_95, ap50, summary = COCOEvaluator(json_list, self.trainer.datamodule.dataset_test)
    #     print("Batch {:d}, mAP = {:.3f}, mAP50 = {:.3f}".format(self.current_epoch, ap50_95, ap50))
    #     print(summary)
    #     # VOC Evaluator
    #     # VOCEvaluator(det_list, self.trainer.datamodule.dataset_test, iou_thr=0.5)
    #     # inference time
    #     average_ifer_time = torch.tensor(self.infr_times, dtype=torch.float32).mean().item()
    #     average_nms_time = torch.tensor(self.nms_times, dtype=torch.float32).mean().item()
    #     print("The average iference time is %.4fs, nms time is %.4fs" % (average_ifer_time, average_nms_time))
