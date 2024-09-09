import torch


def npu_linear(x, weight, bias):
    output = torch.nn.functional.linear(x, weight, bias)
    return output


def npu_linear_backward(grad, input_data, weight):
    input_grad = torch.matmul(grad, weight)
    weight_grad = torch.matmul(grad.t(), input_data)
    return input_grad.cpu(), weight_grad.cpu()
