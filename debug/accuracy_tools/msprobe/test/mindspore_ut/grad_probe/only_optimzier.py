import mindspore
from mindspore import nn, Tensor
from mindspore.nn import SGD
import numpy as np
import os

file_path = os.path.abspath(__file__)
directory = os.path.dirname(file_path)

from grad_tool.grad_monitor import GradientMonitor


def main():
    gm = GradientMonitor(os.path.join(directory, "config.yaml"), framework="MindSpore")

    class SimpleNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.my_dense = nn.Dense(16, 5)

        def construct(self, x):
            x = self.flatten(x)
            logits = self.my_dense(x)
            return logits
    model = SimpleNet()
    optimizer = SGD(model.trainable_params(), learning_rate=0.001)

    gm.monitor(optimizer)

    fix_gradient = tuple([Tensor(np.arange(5*16).reshape((5, 16)), dtype=mindspore.float32),
                        Tensor(np.arange(5).reshape(5), dtype=mindspore.float32)])

    steps = 10

    for _ in range(steps):
        optimizer(fix_gradient)

if __name__ == "__main__":
    main()
