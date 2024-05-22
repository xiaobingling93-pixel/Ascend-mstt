import torch

_recursive_key_stack = []
def recursive_apply_transform(args, transform):
    global _recursive_key_stack
    if isinstance(args, (list, tuple)):
        transform_result = []
        for i, arg in enumerate(args):
            _recursive_key_stack.append(str(i))
            transform_result.append(recursive_apply_transform(arg, transform))
            _recursive_key_stack.pop()
        return type(args)(transform_result)
    elif isinstance(args, dict):
        transform_result = {}
        for k, arg in args.items():
            _recursive_key_stack.append(str(k))
            transform_result[k] = recursive_apply_transform(arg, transform)
            _recursive_key_stack.pop()
        return transform_result
    else:
        arg_transform = transform(args, _recursive_key_stack)
        return arg_transform

