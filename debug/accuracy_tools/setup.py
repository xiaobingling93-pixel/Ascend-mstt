# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
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

import setuptools


__version__ = '1.1.0'

import setuptools

INSTALL_REQUIRED = [
    "wheel",
    "einops",
    "numpy < 2.0",
    "pandas >= 1.3.5, < 2.1",
    "pyyaml",
    "rich",
    "tqdm",
    "openpyxl",
    "pyopenssl",
    "twisted",
    "matplotlib"
]

EXCLUDE_PKGS = [
    "api_accuracy_checker*",
    "grad_tool*",
    "kj600*",
    "ptdbg_ascend*",
    "msprobe.test*",
]

setuptools.setup(
    name="mindstudio-probe",
    version=__version__,
    description="Pytorch Ascend Probe Utils",
    long_description="MindStudio-Probe is a set of tools for diagnosing and improving model accuracy on Ascend NPU, "
                     "including API acc checker, ptdbg, grad tool etc.",
    url="https://gitee.com/ascend/mstt/tree/master/debug/accuracy_tools/msprobe",
    author="Ascend Team",
    author_email="pmail_mindstudio@huawei.com",
    packages=setuptools.find_namespace_packages(exclude=EXCLUDE_PKGS, include=["msprobe", "msprobe*"]),
    include_package_data=True,
    python_requires=">=3.6.2",
    install_requires=INSTALL_REQUIRED,
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache License 2.0',
    keywords='pytorch msprobe ascend',
    ext_modules=[],
    zip_safe=False,
    entry_points={
        'console_scripts': ['msprobe=msprobe.msprobe:main'],
    }, )
