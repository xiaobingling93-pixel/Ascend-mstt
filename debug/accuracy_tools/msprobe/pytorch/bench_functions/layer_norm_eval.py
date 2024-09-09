import torch


def npu_layer_norm_eval(data, normalized_shape, weight=None, bias=None, eps=1e-5):
    result = torch.nn.functional.layer_norm(data, normalized_shape)
    return result
