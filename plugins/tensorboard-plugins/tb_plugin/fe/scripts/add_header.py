# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.
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
# !/usr/bin/env python
import glob
import os
import sys

HEADER = '''/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

'''


def add_header(file):
    with open(file, 'r') as f:
        contents = f.readlines()

    # do nothing if there is already header
    if contents and contents[0].startswith('/*-'):
        return

    with open(file, 'w') as out:
        out.write(HEADER)
        out.writelines(contents)


if __name__ == '__main__':
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        raise ValueError('{} is not a directory'.format(directory))

    for ts_file in glob.glob(directory + '/*.ts'):
        add_header(ts_file)
