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

from msprof_analyze.prof_exports.base_stats_export import BaseStatsExport
from msprof_analyze.prof_common.constant import Constant

QUERY = """
SELECT
    NAME_IDS.value AS "OpName",
    TYPE_IDS.value AS "OpType",
    round(endNs - startNs) AS "Duration",
    GROUP_NAME_IDS.value AS "GroupName"
FROM
    COMMUNICATION_OP
LEFT JOIN
    STRING_IDS AS TYPE_IDS
    ON TYPE_IDS.id = COMMUNICATION_OP.opType
LEFT JOIN
    STRING_IDS AS NAME_IDS
    ON NAME_IDS.id = COMMUNICATION_OP.opName
LEFT JOIN
    STRING_IDS AS GROUP_NAME_IDS
    ON GROUP_NAME_IDS.id = COMMUNICATION_OP.groupName
{}
    """


class HcclSumExport(BaseStatsExport):

    def __init__(self, db_path, recipe_name, step_range):
        super().__init__(db_path, recipe_name, step_range)
        filter_stat = "WHERE COMMUNICATION_OP.startNs >= ? and COMMUNICATION_OP.startNs <= ?" if step_range else ""
        self._query = QUERY.format(filter_stat)
