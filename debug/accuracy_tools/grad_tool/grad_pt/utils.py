import os
import torch
import torch.distributed as dist


def get_rank_id():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return os.getpid()


def print_rank_0(message):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message)
    else:
        print(message)

class GradConst:
    md5 = "MD5"
    distribution = "distribution"
    shape = "shape"
    max = "max"
    min = "min"
    norm = "norm"
