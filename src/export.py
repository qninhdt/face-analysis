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
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize

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
    model.eval()

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
            "input": {0: "batch_size", 1: "channels", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "num_detections", 3: "data"},
        },
    )

    quantized_model = quantize_dynamic(
        cfg.onnx_path, cfg.quantized_onnx_path, weight_type=QuantType.QUInt8
    )


if __name__ == "__main__":
    main()
