import torch


def npu_rotary_mul(x, r1, r2):
    x1, x2 = torch.chunk(x, 2, -1)
    x_new = torch.cat((-x2, x1), dim=-1)
    output = r1 * x + r2 * x_new
    return output


def npu_rotary_mul_backward(dy_tensor, x, r1, r2):
    x.requires_grad = True
    r1.requires_grad = True
    r2.requires_grad = True
    # golden
    x1, x2 = torch.chunk(x, 2, -1)
    x_new = torch.cat((-x2, x1), dim=-1)
    golden_tensor = r1 * x + r2 * x_new
    golden_tensor.backward(dy_tensor)
    r1_shape = r1.shape
    r1_grad = torch.zeros(r1_shape).type(torch.float32)
    r2_grad = torch.zeros(r1_shape).type(torch.float32)
    x1, x2 = torch.chunk(x.float(), 2, -1)
    x_new2 = torch.cat((-x2, x1), dim=-1)
    x_shape = x.shape
    h = x.float()
    grad = dy_tensor.float()
    condition_1 = (((r1_shape[0] == 1 and x_shape[0] != 1) or (r1_shape[0] == 1 and x_shape[0] == 1)) and
                   ((r1_shape[2] == 1 and x_shape[2] != 1) or (r1_shape[2] == 1 and x_shape[2] == 1)) and
                   (r1_shape[1] == x_shape[1]) and (r1_shape[3] == x_shape[3]))
    condition_2 = (((r1_shape[0] == 1 and x_shape[0] != 1) or (r1_shape[0] == 1 and x_shape[0] == 1)) and
                   ((r1_shape[1] == 1 and x_shape[1] != 1) or (r1_shape[1] == 1 and x_shape[1] == 1)) and
                   (r1_shape[2] == x_shape[2]) and (r1_shape[3] == x_shape[3]))
    condition_3 = (((r1_shape[2] == 1 and x_shape[2] != 1) or (r1_shape[2] == 1 and x_shape[2] == 1)) and
                   ((r1_shape[1] == 1 and x_shape[1] != 1) or (r1_shape[1] == 1 and x_shape[1] == 1)) and
                   (r1_shape[0] == x_shape[0]) and (r1_shape[3] == x_shape[3]))
    if condition_1:
        for i in range(x_shape[0]):
            for j in range(x_shape[2]):
                r2_grad[0, :, 0, :] += (x_new2[i, :, j, :] * grad[i, :, j, :])
                r1_grad[0, :, 0, :] += (h[i, :, j, :] * grad[i, :, j, :])
    elif condition_2:
        for i in range(x_shape[0]):
            for j in range(x_shape[1]):
                r2_grad[0, 0, :, :] += (x_new2[i, j, :, :] * grad[i, j, :, :])
                r1_grad[0, 0, :, :] += (h[i, j, :, :] * grad[i, j, :, :])
    elif condition_3:
        for i in range(x_shape[1]):
            for j in range(x_shape[2]):
                r2_grad[:, 0, 0, :] += (x_new2[:, i, j, :] * grad[:, i, j, :])
                r1_grad[:, 0, 0, :] += (h[:, i, j, :] * grad[:, i, j, :])
    return x.grad.cpu(), r1_grad.cpu(), r2_grad.cpu()
