import torch
import numpy as np

special_type = (torch.device, torch.dtype, torch.Size, torch.Tensor, np.integer, np.floating, np.bool_, np.complexfloating, \
                np.str_, np.byte, np.unicode_, bool, int, float, str, slice)
def recursive_apply_transform(args, transform):
    if isinstance(args, special_type):
        arg_transform = transform(args)
        return arg_transform
    elif isinstance(args, (list, tuple)):
        transform_result = []
        for i, arg in enumerate(args):
            transform_result.append(recursive_apply_transform(arg, transform))
        return type(args)(transform_result)
    elif isinstance(args, dict):
        transform_result = {}
        for k, arg in args.items():
            transform_result[k] = recursive_apply_transform(arg, transform)
        return transform_result
