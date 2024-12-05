import mindspore as ms
from mindspore import Tensor, ops

ORIGIN_GRAD = ops.composite.base._Grad

class CustomGradOperation:
    def custom_grad_fn(self, gradients):
        for idx, grad in enumerate(gradients):
            ops.TensorDump()(rf"dump/grad", grad)
        return gradients

    def __init__(self, *args, **kwargs):
        """Initialize CustomGradOperation"""
        super().__init__()
        self.grad = ops.GradOperation(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        grad_fn = self.grad(*args, **kwargs)

        def wrapped_grad_fn(*args, **kwargs):
            gradients = grad_fn(*args, **kwargs)
            self.custom_grad_fn(gradients)
            return gradients
        return wrapped_grad_fn


class CustomLiteGradOperation:
    def custom_grad_fn(self, gradients):
        for idx, grad in enumerate(gradients):
            ops.TensorDump()(rf"dump/grad", grad)
        return gradients

    def __init__(self, *args, **kwargs):
        """Initialize CustomGradOperation"""
        super().__init__()
        self.grad = ORIGIN_GRAD(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        grad_fn = self.grad(*args, **kwargs)

        def wrapped_grad_fn(*args, **kwargs):
            gradients = grad_fn(*args, **kwargs)
            self.custom_grad_fn(gradients)
            return gradients
        return wrapped_grad_fn


def enable_hook():
    ops.composite.GradOperation = CustomGradOperation
    ops.composite.base._Grad = CustomLiteGradOperation