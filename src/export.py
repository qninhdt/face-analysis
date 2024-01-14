import hydra
import torch
from lightning import LightningModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import os
import sys
import numpy as np
import torch.onnx
from pathlib import Path
import torch.quantization

sys.path.append(str(Path(__file__).parent))
os.environ["PROJECT_ROOT"] = str(Path(__file__).parent.parent)

from utils import (
    RankedLogger,
)

log = RankedLogger(__name__, rank_zero_only=True)


def get_model_size(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    return size


@hydra.main(version_base="1.3", config_path="../configs", config_name="export.yaml")
def main(cfg: DictConfig) -> None:
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])

    model = model.model

    log.info(f"Model size: {get_model_size(model):.2f} MB")

    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)

    log.info(f"Quantized model size: {get_model_size(model):.2f} MB")

    print(model)

    torch.onnx.export(
        model,
        torch.randn(1, 3, 640, 640),
        cfg.onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    main()
