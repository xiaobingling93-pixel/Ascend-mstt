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
from msprof_analyze.compare_tools.compare_backend.comparator.base_comparator import BaseComparator


class OperatorComparator(BaseComparator):
    def __init__(self, origin_data: any, bean: any):
        super().__init__(origin_data, bean)

    def _compare(self):
        if not self._origin_data:
            return
        self._rows = [None] * (len(self._origin_data))
        for index, (base_op, comparison_op) in enumerate(self._origin_data):
            self._rows[index] = self._bean(index, base_op, comparison_op).row
