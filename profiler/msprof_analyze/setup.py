#!/usr/bin/python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#    http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from setuptools import find_packages, setup  # type: ignore

from msprof_analyze.prof_common.path_manager import PathManager
from msprof_analyze.prof_common.file_manager import FileManager
from msprof_analyze.prof_common.utils import SafeConfigReader

extras = {
    "test": [
        "pytest==6.2.4",
        "pytest-cookies==0.6.1",
        "pytest-cov==2.12.0",
        "mock==4.0.3",
    ]
}

sections = {
    'URL': ['msprof_analyze_url'],
    'EMAIL': ['ms_email']
}

requires = FileManager.read_common_file('requirements/build.txt').splitlines()

tests_requires = FileManager.read_common_file('requirements/tests.txt').splitlines()
tests_requires.extend(set(requires))

version = FileManager.read_common_file('version.txt').strip()

CONFIG_FILE_PATH = "config/config.ini"
PathManager.check_input_file_path(CONFIG_FILE_PATH)
PathManager.check_file_size(CONFIG_FILE_PATH)
reader = SafeConfigReader(CONFIG_FILE_PATH)
reader.validate(sections)
config = reader.get_config()
try:
    url = config.get("URL", "msprof_analyze_url")
except Exception as e:
    raise RuntimeError("The configuration file is incomplete and not configured msprof_analyze_url information.") from e
try:
    author_email = config.get("EMAIL", "ms_email")
except Exception as e:
    raise RuntimeError("The configuration file is incomplete and not configured ms_email information.") from e

root_path = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
msprof_analyze_path = os.path.abspath(os.path.dirname(__file__))
child_packages = find_packages(msprof_analyze_path, exclude=["example"])
msprof_analyze_packages = [f"msprof_analyze.{package}" for package in child_packages]
setup(
    name="msprof-analyze",
    version=version,
    description="MindStudio Profiler Analyze Tools",
    long_description="msprof-analyze provides statistics, analysis, and related tuning suggestions for the "
                     "performance data collected in training and large model scenarios. The main functional modules"
                     " include: performance comparison, performance analysis, and cluster analysis.",
    url=url,
    author="MindStudio",
    author_email=author_email,
    package_dir={"": root_path,
                 "msprof_analyze": msprof_analyze_path},
    packages=find_packages(root_path, exclude=["example"]) + msprof_analyze_packages,
    include_package_data=False,
    python_requires='>=3.7',
    install_requires=requires,
    package_data={'': ['*.json', '*.ini', '*.txt', '*.yaml', '*.html', '*.ipynb']},
    tests_require=tests_requires,
    license='Apache License 2.0',
    entry_points="""
        [console_scripts]
        msprof-analyze=msprof_analyze.cli.entrance:msprof_analyze_cli
    """
)

# build cmd: pip install --editable .
