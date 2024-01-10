# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
#
# Copyright(c) 2023 Huawei Technologies.
# All rights reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications: Add visualization of PyTorch Ascend profiling.
# --------------------------------------------------------------------------
import os
import pathlib
import subprocess
import setuptools


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            version = line.split(delim)[1]

    if os.getenv('TORCH_TB_PROFILER_BUILD_VERSION'):
        version = os.getenv('TORCH_TB_PROFILER_BUILD_VERSION')
    return version


INSTALL_REQUIRED = [
    "pandas >= 1.0.0",
    "tensorboard >= 2.11.0"
]

TESTS_REQUIRED = INSTALL_REQUIRED + [
    "torch >= 1.8",
    "torchvision >= 0.8"
]

EXTRAS = {
    "s3": ["boto3"],
    "blob": ["azure-storage-blob"],
    "gs": ["google-cloud-storage"]
}


class build_fe(setuptools.Command):
    """Build the frontend"""
    description = "run yarn build on frontend directory"

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        cwd = pathlib.Path().absolute()
        root = pathlib.Path(__file__).parent.absolute()
        os.chdir(root / "fe")
        subprocess.run(["yarn", "build:copy"], check=True)
        # restore the working directory
        os.chdir(cwd)


setuptools.setup(
    name="torch-tb-profiler-ascend",
    version=get_version(os.path.join('torch_tb_profiler', '__init__.py')),
    description="PyTorch Ascend Profiler TensorBoard Plugin",
    long_description="PyTorch Ascend Profiler TensorBoard Plugin : \
        https://gitee.com/ascend/att/tree/master/plugins/tensorboard-plugins/tb_plugin",
    url="https://gitee.com/ascend/att/tree/master/plugins/tensorboard-plugins/tb_plugin",
    author="Ascend Team",
    author_email="pmail_mindstudio@huawei.com",
    cmdclass={
        "build_fe": build_fe
    },
    packages=setuptools.find_packages(),
    package_data={
        "torch_tb_profiler": ["static/**"],
    },
    entry_points={
        "tensorboard_plugins": [
            "torch_profiler = torch_tb_profiler.plugin:TorchProfilerPlugin",
        ],
    },
    python_requires=">=3.6.2",
    install_requires=INSTALL_REQUIRED,
    tests_require=TESTS_REQUIRED,
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='BSD-3',
    keywords='pytorch tensorboard profile plugin',
    extras_require=EXTRAS
)
