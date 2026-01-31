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


import os

import numpy as np


class Const:
    MS_ACCU_CMP_PATH = '/usr/local/Ascend/ascend-toolkit/latest/tools/operator_cmp/compare/msaccucmp.py'
    MS_ACCU_CMP_FILE_NAME = 'msaccucmp.py'
    ROOT_DIR = ""
    LOG_LEVEL = "NOTSET"
    DATA_ROOT_DIR = os.path.join(ROOT_DIR, 'parse_data')
    DUMP_CONVERT_DIR = os.path.join(DATA_ROOT_DIR, 'dump_convert')
    COMPARE_DIR = os.path.join(DATA_ROOT_DIR, 'compare_result')
    BATCH_DUMP_CONVERT_DIR = os.path.join(DATA_ROOT_DIR, 'batch_dump_convert')
    BATCH_COMPARE_DIR = os.path.join(DATA_ROOT_DIR, 'batch_compare_result')
    OFFLINE_DUMP_CONVERT_PATTERN = \
        r"^([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)\.([0-9]+)(\.[0-9]+)?\.([0-9]{1,255})" \
        r"\.([a-z]+)\.([0-9]{1,255})(\.[x0-9]+)?\.npy$"
    NUMPY_PATTERN = r"^[\w\-_.]+\.npy$"
    NPY_SUFFIX = ".npy"
    PKL_SUFFIX = ".pkl"
    DIRECTORY_LENGTH = 4096
    FILE_NAME_LENGTH = 255
    MAX_TRAVERSAL_DEPTH = 5
    FILE_PATTERN = r'^[a-zA-Z0-9_./-]+$'
    ONE_GB = 1 * 1024 * 1024 * 1024
    TEN_GB = 10 * 1024 * 1024 * 1024
    FLOAT_TYPE = [np.half, np.single, float, np.double, np.float64, np.longdouble, np.float32, np.float16]
    HEADER = r"""    ____
       / __ \____ ______________
      / /_/ / __ `/ ___/ ___/ _ \
     / ____/ /_/ / /  (__  )  __/
    /_/    \__,_/_/  /____/\___/

    """
