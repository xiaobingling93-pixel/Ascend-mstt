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


from msprobe.mindspore.common.const import FreeBenchmarkConst


class Config:
    is_enable: bool = False
    handler_type = FreeBenchmarkConst.DEFAULT_HANDLER_TYPE
    pert_type = FreeBenchmarkConst.DEFAULT_PERT_TYPE
    stage = FreeBenchmarkConst.DEFAULT_STAGE
    dump_level = FreeBenchmarkConst.DEFAULT_DUMP_LEVEL
    steps: list = []
    ranks: list = []
    dump_path: str = ""
