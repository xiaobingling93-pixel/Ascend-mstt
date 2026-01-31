# -------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

from msprof_analyze.prof_common.db_manager import DBManager
from msprof_analyze.prof_common.constant import Constant
from msprof_analyze.prof_exports.base_stats_export import BaseStatsExport


class PPChartExport(BaseStatsExport):
    QUERY = """
    SELECT
        {}
        MSG_IDS.value AS msg,
        TASK.startNs,
        TASK.endNs
    FROM
        MSTX_EVENTS
    JOIN
        TASK ON MSTX_EVENTS.connectionId = TASK.connectionId
    JOIN
        STRING_IDS AS MSG_IDS ON MSTX_EVENTS.message = MSG_IDS.id
    {}
    WHERE
        msg LIKE '%forward%'
        OR msg LIKE '%backward%'
        OR msg LIKE '%WeightGradStore_pop%'
        {}
    ORDER BY
        TASK.startNs
    """

    def __init__(self, db_path, recipe_name, step_range):
        super().__init__(db_path, recipe_name, step_range)
        self._query = self._build_query(db_path, step_range)

    def _build_query(self, db_path, step_range):
        str1 = "0 AS step,"
        str2 = ""
        filter_statement = "AND MSTX_EVENTS.startNs >= ? and MSTX_EVENTS.startNs <= ?" if step_range else ""
        if DBManager.check_tables_in_db(db_path, Constant.TABLE_STEP_TIME):
            str1 = "step_time.id AS step,"
            str2 = """
            LEFT JOIN STEP_TIME step_time
                ON MSTX_EVENTS.startNs >= step_time.startNs
                AND MSTX_EVENTS.endNs <= step_time.endNs
            """
        return self.QUERY.format(str1, str2, filter_statement)
