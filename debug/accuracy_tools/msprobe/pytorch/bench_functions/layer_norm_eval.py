import torch


def npu_layer_norm_eval(data, normalized_shape):
    result = torch.nn.functional.layer_norm(data, normalized_shape)
    return result.cpu()
