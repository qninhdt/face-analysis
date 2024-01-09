from typing import Any, Dict, List, Tuple
import hydra
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import os
import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from model.evaluators.postprocess import postprocess

sys.path.append(str(Path(__file__).parent))
os.environ['PROJECT_ROOT'] = str(Path(__file__).parent.parent)

from utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def submit(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    assert cfg.ckpt_path

    log.info(f"Instantiating dataset <{cfg.data._target_}>")
    dataset: Dataset = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting inferencing!")

    labels_map = {
        "age": [
            "Baby",
            "Kid",
            "Teenager",
            "20-30s",
            "40-50s",
            "Senior"
        ],
        "race": [
            "Caucasian",
            "Mongoloid",
            "Negroid"
        ],
        "masked": [
            "unmasked",
            "masked"
        ],
        "skintone": [
            "light",
            "mid-light",
            "mid-dark",
            "dark"
        ],
        "emotion": [
            "Neutral",
            "Happiness",
            "Surprise",
            "Sadness",
            "Fear",
            "Disgust",
            "Anger"
        ],
        "gender": [
            "Male",
            "Female"
        ]
    }

    def collate_fn(batch):
        imgs = torch.stack([sample[0] for sample in batch])
        labels = [sample[1] for sample in batch]

        return imgs, labels

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=2,
        num_workers=8,
        shuffle=False,
        collate_fn=collate_fn
    )

    # for predictions use trainer.predict(...)
    outputs = trainer.predict(model=model, dataloaders=data_loader, ckpt_path=cfg.ckpt_path)

    preds = []
    for predictions, labels in outputs:
        image_sizes = [[label['width'], label['height']] for label in labels]
        pred = postprocess(predictions, origin_sizes=image_sizes)
        preds.extend(pred)

    print(f"Finish inferencing {len(preds)} images")

    results = []
    image_list = dataset.image_list

    for i, pred in enumerate(preds):
        for j in range(pred['boxes'].shape[0]):

            # convert cxcywh to xywh
            pred['boxes'][j][0] -= pred['boxes'][j][2] / 2
            pred['boxes'][j][1] -= pred['boxes'][j][3] / 2

            result = {
                "image_id": image_list[i][1],
                "file_name": image_list[i][0],
                "bbox": str(pred['boxes'][j].tolist()),
                "age": labels_map['age'][pred['age'][j].long().item()],
                "race": labels_map['race'][pred['race'][j].long().item()],
                "skintone": labels_map['skintone'][pred['skintone'][j].long().item()],
                "emotion": labels_map['emotion'][pred['emotion'][j].long().item()],
                "gender": labels_map['gender'][pred['gender'][j].long().item()],
                "masked": labels_map['masked'][pred['masked'][j].long().item()],
            }

            results.append(result)

    # convert to csv
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("answer.csv", index=False)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    submit(cfg)


if __name__ == "__main__":
    main()
