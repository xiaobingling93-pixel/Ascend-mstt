# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
# `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


__version__ = '8.3.3'

import os
import subprocess
import platform
import sys
import setuptools


INSTALL_REQUIRED = [
    "wheel",
    "einops",
    "numpy >=1.23.0, < 2.0",
    "pandas >= 1.3.5, < 2.1",
    "pyyaml",
    "rich",
    "tqdm",
    "openpyxl >= 3.0.6",
    "pyopenssl==24.2.1",
    "twisted",
    "matplotlib",
    "tensorboard",
    "tabulate",
    "pwinput",
    "psutil"
]

EXCLUDE_PKGS = [
    "api_accuracy_checker*",
    "grad_tool*",
    "ptdbg_ascend*",
    "msprobe.ccsrc*",
    "msprobe.test*",
    "build.sh",
    "build_dependency*",
    "cmake*",
    "output*",
    "third_party*",
]

if "--plat-name" in sys.argv or "--python-tag" in sys.argv:
    raise SystemError("Specifing platforms or python version is not supported.")

if platform.system() != "Linux":
    raise SystemError("MsProbe is only supported on Linux platforms.")

mod_list_range = {"adump", }
mod_list = []
for i, arg in enumerate(sys.argv):
    if arg.startswith("--include-mod"):
        if "--no-check" in sys.argv:
            os.environ["INSTALL_WITHOUT_CHECK"] = "1"
            sys.argv.remove("--no-check")
        if arg.startswith("--include-mod="):
            mod_list = arg[len("--include-mod="):].split(',')
            sys.argv.remove(arg)
        elif i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
            mod_list = sys.argv[i + 1].split(',')
            sys.argv.remove(sys.argv[i + 1])
            sys.argv.remove(arg)
        mod_list = list(set(mod_list) & mod_list_range)
        break

# 当前只有adump一个mod
if mod_list:
    arch = platform.machine()
    sys.argv.append("--plat-name")
    sys.argv.append(f"linux_{arch}")
    sys.argv.append("--python-tag")
    sys.argv.append(f"cp{sys.version_info.major}{sys.version_info.minor}")
    build_cmd = f"bash ./build.sh -j16 -a {arch} -v {sys.version_info.major}.{sys.version_info.minor}"
    p = subprocess.run(build_cmd.split(), shell=False)
    if p.returncode != 0:
        raise RuntimeError(f"Failed to build source({p.returncode})")

setuptools.setup(
    name="mindstudio-probe",
    version=__version__,
    description="Ascend Probe Utils",
    long_description="MindStudio-Probe is a set of tools for diagnosing and improving model accuracy on Ascend NPU, "
                     "including API acc checker, ptdbg, grad tool etc.",
    url="https://gitcode.com/Ascend/mstt/tree/master/debug/accuracy_tools/msprobe",
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
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Mulan PSL v2',
    keywords='pytorch msprobe ascend',
    ext_modules=[],
    zip_safe=False,
    entry_points={
        'console_scripts': ['msprobe=msprobe.msprobe:main'],
    }, )
