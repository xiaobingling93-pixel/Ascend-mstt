import torch


def npu_swiglu(x, dim=-1):
    tensor_dtype = x.dtype

    inTensors = torch.chunk(x, 2, dim=dim)
    if tensor_dtype == torch.float32:
        tensor_scalar = torch.sigmoid(torch.mul(inTensors[0], 1.0))
        output_data = torch.mul(torch.mul(tensor_scalar, inTensors[0]), inTensors[1])
    else:
        tensor_self_float = inTensors[0].type(torch.float)
        tensor_other_float = inTensors[1].type(torch.float)
        tensor_out_float = torch.nn.functional.silu(tensor_self_float).type(tensor_dtype).type(
            torch.float32) * tensor_other_float
        output_data = tensor_out_float.type(tensor_dtype)
    return output_data


def npu_swiglu_backward(grad, x, dim=-1):
    tensor_dtype = grad.dtype
    in_tensors = torch.chunk(x, 2, dim=dim)
    tensor_grad_out = grad

    if tensor_dtype == torch.float16:
        tensor_out1 = torch.mul(
            torch.mul(in_tensors[1].type(torch.float32), swish_grad(1, in_tensors[0].type(torch.float32))),
            tensor_grad_out.type(torch.float32)).type(torch.float16)
        tensor_out2 = torch.mul(tensor_grad_out.type(torch.float32),
                                swish(1, in_tensors[0].type(torch.float32))).type(torch.float16)
        output = torch.cat((tensor_out1, tensor_out2), dim)
    elif tensor_dtype == torch.bfloat16:
        tensor_self_float = in_tensors[0].type(torch.float)
        tensor_other_float = in_tensors[1].type(torch.float)
        tensor_gradout_float = tensor_grad_out.type(torch.float)

        tensor_out1 = torch.mul(tensor_gradout_float, swish_grad(1.0, tensor_self_float)).type(torch.bfloat16).type(
            torch.float32) * tensor_other_float
        tensor_out2 = swish(1.0, tensor_self_float).type(torch.bfloat16).type(torch.float32) * tensor_gradout_float
        tensor_out_float = torch.cat((tensor_out1, tensor_out2), dim=dim)
        output = tensor_out_float.type(torch.bfloat16)
    else:
        tensor_out1 = torch.mul(torch.mul(in_tensors[1], swish_grad(1.0, in_tensors[0])), tensor_grad_out)
        tensor_out2 = torch.mul(tensor_grad_out, swish(1.0, in_tensors[0]))
        output = torch.cat((tensor_out1, tensor_out2), dim)
    return output.cpu()


def swish_grad(beta, x):
    return torch.sigmoid(beta * x) + x * (1 - torch.sigmoid(beta * x)) * torch.sigmoid(beta * x) * beta


def swish(beta, x):
    return x * torch.sigmoid(beta * x)

