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


from msprobe.core.common.utils import CompareException
from msprobe.core.common.file_utils import create_directory
from msprobe.core.common.exceptions import FileCheckException
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.compare.ms_compare import ms_compare
from msprobe.core.compare.utils import compare_distributed_inner
from msprobe.mindspore.compare.ms_graph_compare import GraphMSComparator


def ms_compare_distributed(npu_dump_dir, bench_dump_dir, output_path, **kwargs):
    compare_distributed_inner(npu_dump_dir, bench_dump_dir, output_path, ms_compare, **kwargs)


def ms_graph_compare(inputs, outputs):
    try:
        create_directory(outputs)
    except (CompareException, FileCheckException) as error:
        logger.error('Compare failed. Please check the arguments and do it again!')
        return
    ms_comparator = GraphMSComparator(inputs, outputs)
    ms_comparator.compare_core()
