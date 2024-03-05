import os
import subprocess
import torch
import torch_npu
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
CUR_PATH = os.path.abspath(os.path.dirname(__file__))


def compile_kernels():
    # 由于pytorch中没有昇腾device编译的扩展，所以此处人工加make
    subprocess.run("make")
    return "libcustom_kernels.so"  # 这个make出来的库名字


def compile_adapter():
    ext = NpuExtension(
        name="ascend_custom_kernels_lib",  # import的库的名字
        # 如果还有其他cpp文件参与编译，需要在这里添加
        sources=[f"{CUR_PATH}/add_adapter.cpp"],
        extra_compile_args=[
            '-I' + os.path.join(os.path.join(os.path.join(os.path.join(
                PYTORCH_NPU_INSTALL_PATH, "include"), "third_party"), "acl"), "inc"),
        ],
        library_dirs=[f"{CUR_PATH}"],  # 编译时需要依赖的库文件的路径，相当于g++编译时的-L选项
        libraries=["custom_kernels"],  # 编译时依赖的库文件，相当于-l选项
    )
    return [ext]


if __name__ == "__main__":
    # 编译出含有算子的库，并以so的方式提供
    kernel_so = compile_kernels()

    # 编译出pytorch适配层的库，支持被框架集成
    exts = compile_adapter()

    # 将整体打包成wheel包
    setup(
        name="ascend_custom_kernels",  # package的名字
        version='1.0',
        keywords='ascend_custom_kernels',
        ext_modules=exts,
        packages=find_packages(),
        cmdclass={"build_ext": BuildExtension},
        data_files=[(".", [kernel_so])],
        include_package_data=True,
    )
