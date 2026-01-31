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

QUERY = """
SELECT
    ta.startNs,
    ta.endNs,
    ta.connectionId,
    si.value
FROM
    MSTX_EVENTS ms
JOIN
    TASK ta
    ON ms.connectionId = ta.connectionId
JOIN
    STRING_IDS si
    ON ms.message = si.id
WHERE
    si.value LIKE '%"streamId":%'
    AND si.value LIKE '%"count":%'
    AND si.value LIKE '%"dataType":%'
    AND si.value LIKE '%"groupName":%'
    AND si.value LIKE '%"opName":%'
    {}
    """


class Mstx2CommopExport(BaseStatsExport):

    def __init__(self, db_path, recipe_name, step_range):
        super().__init__(db_path, recipe_name, step_range)
        filter_stat = "AND ms.startNs >= ? and ms.startNs <= ?" if step_range else ""
        self._query = QUERY.format(filter_stat)