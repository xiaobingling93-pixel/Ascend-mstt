import torch
from api_accuracy_checker.common.function_factory import npu_custom_functions, npu_custom_grad_functions


@npu_custom_functions
def npu_confusion_transpose(data, perm, shape, transpose_first):
    if transpose_first:
        output = data.permute(*perm).contiguous().view(shape)
    else:
        output = data.view(shape).permute(*perm)
    return output.cpu()


@npu_custom_grad_functions
def npu_confusion_transpose_backward(grad, perm, shape, transpose_first):
    shape_cal = shape if transpose_first else [shape[perm_dim] for perm_dim in perm]
    perm_cal = [0] * len(perm)
    for i, perm_dim in enumerate(perm):
        perm_cal[perm_dim] = i

    if transpose_first:
        result = grad.permute(*perm_cal).reshape(shape_cal)
    else:
        result = grad.reshape(shape_cal).permute(*perm_cal)
    return result.cpu()
