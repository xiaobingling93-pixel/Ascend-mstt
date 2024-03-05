import os
import subprocess
import torch
import torch_npu
import torch.utils.cpp_extension
from torch_npu.testing.testcase import TestCase, run_tests

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
CUR_PATH = os.path.abspath(os.path.dirname(__file__))


def compile_kernels():
    # 由于pytorch中没有昇腾device编译的扩展，所以此处人工加make
    subprocess.run("make")


def compile_host():
    extra_ldflags = []
    extra_ldflags.append(f"-L{PYTORCH_NPU_INSTALL_PATH}/lib")
    extra_ldflags.append("-ltorch_npu")
    extra_ldflags.append(f"-L{CUR_PATH}/")
    extra_ldflags.append("-lcustom_kernels")
    extra_include_paths = []
    extra_include_paths.append("./")
    extra_include_paths.append(os.path.join(
        PYTORCH_NPU_INSTALL_PATH, "include"))
    extra_include_paths.append(os.path.join(os.path.join(os.path.join(os.path.join(
        PYTORCH_NPU_INSTALL_PATH, "include"), "third_party"), "acl"), "inc"))

    module = torch.utils.cpp_extension.load(
        name="jit_extension",
        sources=[
            "add_adapter.cpp"
        ],
        extra_include_paths=extra_include_paths,
        extra_ldflags=extra_ldflags,
        verbose=True)
    return module


class TestCustomAdd(TestCase):
    def test_add(self):
        module = compile_host()
        # 由于kernel现在是静态tiling，所以此处尺寸需要匹配
        # 因为add是elementwise的，现有算子支持8*2048(详见kernel实现)，所以，小于这个应该都可以
        length = [8, 2048]
        x = torch.rand(length, device='cpu', dtype=torch.float16)
        y = torch.rand(length, device='cpu', dtype=torch.float16)

        x_npu = x.npu()
        y_npu = y.npu()
        x_npu.requires_grad = True
        y_npu.requires_grad = True
        output = module.my_add(x_npu, y_npu)
        # 反向能力验证
        output.backward(output)

        x.requires_grad = True
        y.requires_grad = True
        cpuout = torch.add(x, y)
        cpuout.backward(cpuout)

        self.assertRtolEqual(output, cpuout)
        self.assertRtolEqual(x_npu.grad, x.grad)
        self.assertRtolEqual(y_npu.grad, y.grad)


if __name__ == '__main__':
    compile_kernels()
    run_tests()
