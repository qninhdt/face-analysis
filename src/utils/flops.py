import torch
from thop import profile, clever_format


def model_summary(model, image_size, device):
    dummy_input = torch.zeros(1, 3, image_size, image_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,))
    flops, params = clever_format([flops*2, params], "%.3f")
    print(" ------- params: %s ------- flops: %s" % (params, flops))
