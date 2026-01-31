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
import re

import pandas as pd

from msprof_analyze.prof_common.db_manager import DBManager
from msprof_analyze.prof_common.constant import Constant
from msprof_analyze.prof_common.logger import get_logger

logger = get_logger()


class BaseStatsExport:

    def __init__(self, db_path, analysis_class, step_range):
        self._db_path = db_path
        self._analysis_class = analysis_class
        self._step_range = step_range
        self._query = None
        self._param = (self._step_range.get(Constant.START_NS),
                       self._step_range.get(Constant.END_NS)) if self._step_range else None

    def get_query(self):
        return self._query

    def read_export_db(self):
        try:
            if not self._db_path:
                logger.error("db path is None.")
                return None
            query = self.get_query()
            if query is None:
                logger.error("query is None.")
                return None
            conn, cursor = DBManager.create_connect_db(self._db_path, Constant.ANALYSIS)
            if self._param is not None and re.search(Constant.SQL_PLACEHOLDER_PATTERN, query):
                data = pd.read_sql(query, conn, params=self._param)
            else:
                data = pd.read_sql(query, conn)
            DBManager.destroy_db_connect(conn, cursor)
            return data
        except Exception as e:
            logger.error(f"File {self._db_path} read failed error: {e}")
            return None
