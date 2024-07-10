import numpy as np
import torch

from .log import print_warn_log

_recursive_key_stack = []
special_type = (torch.device, torch.dtype, torch.Size, torch.Tensor, np.integer, np.floating, np.bool_, np.complexfloating, \
                np.str_, np.byte, np.unicode_, bool, int, float, str, slice)


def recursive_apply_transform(args, transform):
    global _recursive_key_stack
    if isinstance(args, special_type):
        arg_transform = transform(args, _recursive_key_stack)
        return arg_transform
    elif isinstance(args, (list, tuple)):
        transform_result = []
        for i, arg in enumerate(args):
            _recursive_key_stack.append(str(i))
            transform_result.append(recursive_apply_transform(arg, transform))
            _recursive_key_stack.pop()
        return type(args)(transform_result)
    elif isinstance(args, dict):
        transform_dict = {}
        for k, arg in args.items():
            _recursive_key_stack.append(str(k))
            transform_dict[k] = recursive_apply_transform(arg, transform)
            _recursive_key_stack.pop()
        return transform_dict
    elif args is not None:
        print_warn_log(f"Data type {type(args)} is not supported.")
