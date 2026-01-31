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


from msprobe.core.common.const import Const
from msprobe.core.common.log import logger


class DataProcessor:
    def __init__(self, data_frame):
        self.data_frame = data_frame
        if self.data_frame == Const.PT_FRAMEWORK:
            from msprobe.pytorch.compare.distributed_compare import compare_distributed
            self.process_func = compare_distributed
        elif self.data_frame == Const.MS_FRAMEWORK:
            from msprobe.mindspore.compare.distributed_compare import ms_compare_distributed
            self.process_func = ms_compare_distributed
        else:
            raise ValueError(f"Unsupported data_frame: {self.data_frame}")

    def process(self, npu_path, bench_path, output_path):
        logger.info("Start comparing data ......")
        return self.process_func(npu_path, bench_path, output_path, first_diff_analyze=True)
