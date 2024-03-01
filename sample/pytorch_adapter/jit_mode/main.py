import os
import torch
import torch.utils.cpp_extension
import torch_npu

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
CUR_PATH = os.getcwd()


def compile_kernels():
    os.system("make")  # 由于pytorch中没有device编译的扩展，所以此处人工加make


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


def test_add(module):
    # 由于kernel现在是静态tiling，所以此处尺寸需要匹配
    # 因为add是elementwise的，现有算子支持8*2048(详见kernel实现)，所以，小于这个应该都可以
    x = torch.arange(0, 100).short()
    y = torch.arange(0, 100).short()
    z = module.my_add(x.npu(), y.npu())
    print(z)


if __name__ == '__main__':
    compile_kernels()
    module = compile_host()
    test_add(module)
