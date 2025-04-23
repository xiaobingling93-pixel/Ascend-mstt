from mindspore import Tensor
import torch


def create_msa_tensor(data, dtype=None):
    return Tensor(data, dtype)


tensor_tensor = torch.tensor
setattr(torch, 'tensor', create_msa_tensor)


def reset_torch_tensor():
    setattr(torch, 'tensor', tensor_tensor)
