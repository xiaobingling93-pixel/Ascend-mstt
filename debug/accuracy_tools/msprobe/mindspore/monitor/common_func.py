from mindspore import nn
from mindspore import communication
from msprobe.mindspore.monitor.utils import logger
from msprobe.mindspore.common.utils import is_mindtorch
if is_mindtorch():
    import torch


def is_valid_instance(model):
    return isinstance(torch.nn.Module) if is_mindtorch() else isinstance(model, nn.Cell)


def get_submodules(model):
    if not is_valid_instance(model):
        logger.info("Counter invalid model, nothing to hook")
        return {}
    return model.named_modules() if is_mindtorch() else model.cells_and_names()


def get_parameters(model):
    if not is_valid_instance(model):
        return {}
    if is_mindtorch():
        return model.named_parameters()
    else:
        return model.parameters_and_names()


def get_rank():
    if comm_is_initialized():
        return communication.get_rank()
    return 0


def comm_is_initialized():
    return communication.GlobalComm.INITED


def optimizer_pre_hook(optimizer, fn):
    """
    fn should be fn(optimizer, args, **kwargs)
    """
    if is_mindtorch():
        origin_api = optimizer.__class__.step
        def patch_step(func, optimizer):
            def wrapper(*args, **kwargs):
                fn(optimizer, args, kwargs)
                out = func(*args, **kwargs)
                return out
            return wrapper
        optimizer.__class__.step = patch_step(optimizer.__class__.step, optimizer)
        return (optimizer.__class__.step, origin_api)
    else:
        handle = optimizer.register_forward_pre_hook(fn)
        return handle


def optimizer_post_hook(optimizer, fn):
    if is_mindtorch():
        origin_api = optimizer.__class__.step
        def patch_step(func, optimizer):
            def wrapper(*args, **kwargs):
                out = func(*args, **kwargs)
                fn(optimizer, args, kwargs)
                return out
            return wrapper
        optimizer.__class__.step = patch_step(optimizer.__class__.step, optimizer)
        return (optimizer.__class__.step, origin_api)
    else:
        handle = optimizer.register_forward_hook(fn)
        return handle
