import torch
from api_accuracy_checker.common.function_factory import npu_custom_functions, npu_custom_grad_functions


@npu_custom_functions
def npu_rms_norm(x, gamma, eps=1e-5):
    rstd = torch.rsqrt(torch.mean(torch.pow(x, 2), axis=-1, keepdim=True) + eps)
    res = x * rstd * gamma
    return res.cpu(), rstd.cpu()


@npu_custom_grad_functions
def npu_rms_norm_backward(grad, x, gamma, rstd):
    mean_gy = (grad * x * gamma * rstd).mean(dim=-1, keepdim=True)
    grad_x = (grad * gamma - x * rstd * mean_gy) * rstd
    grad_gamma = x * grad * rstd
    return grad_x.cpu(), grad_gamma.cpu()

