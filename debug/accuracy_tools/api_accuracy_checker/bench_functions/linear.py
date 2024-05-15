import torch
from api_accuracy_checker.common.function_factory import npu_custom_functions, npu_custom_grad_functions


@npu_custom_functions
def npu_linear(x, weight, bias):
    output = torch.nn.functional.linear(x, weight, bias)
    return output.cpu()


@npu_custom_grad_functions
def npu_linear_backward(grad, input_data, weight):
    input_grad = torch.matmul(grad, weight)
    weight_grad = torch.matmul(grad.t(), input_data)
    return input_grad.cpu(), weight_grad.cpu()
