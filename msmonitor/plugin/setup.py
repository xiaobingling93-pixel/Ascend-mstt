# Copyright (c) 2025, Huawei Technologies Co., Ltd.
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
import os
from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DYNOLOG_PATH = os.path.join(os.path.dirname(BASE_DIR), "third_party", "dynolog")
GLOG_INC_PATH = os.path.join(DYNOLOG_PATH, "third_party", "glog", "src")
GLOG_LIB_PATH = os.path.join(DYNOLOG_PATH, "build", "third_party", "glog")

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "IPCMonitor",  # Name of the Python module
        sources=["bindings.cpp"] + list(glob("ipc_monitor/*.cpp")), # Source files
        include_dirs=[os.path.join(BASE_DIR, "ipc_monitor"), GLOG_INC_PATH, GLOG_LIB_PATH],  # Include Pybind11 headers
        library_dirs=[GLOG_LIB_PATH],
        extra_compile_args=["-std=c++14", "-fPIC", "-fstack-protector-all", "-fno-strict-aliasing", "-fno-common",
                    "-fvisibility=hidden", "-fvisibility-inlines-hidden", "-Wfloat-equal", "-Wextra", "-O2"],
        libraries=["glog"],
        language="c++",  # Specify the language
    ),
]


# Set up the package
setup(
    name="msmonitor_plugin",
    version="0.1",
    description="msMonitor plugins",
    ext_modules=ext_modules,
    install_requires=["pybind11"],
)