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
from setuptools import setup, find_packages

setup(
    name='mstx_torch_plugin',
    version="1.0",
    description='MindStudio Profiler Mstx Plugin For Pytorch',
    long_description='mstx_torch_plugin provides lightweight data for dataloader, '
                     'forward, step and save_checkpoint.',
    packages=find_packages(),
    license='Apache License 2.0'
)