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
from prettytable import PrettyTable

from msprof_analyze.compare_tools.compare_backend.view.base_view import BaseView


class ScreenView(BaseView):
    def __init__(self, data_dict: dict):
        super().__init__(data_dict)

    def generate_view(self):
        for sheet_name, data in self._data_dict.items():
            if not data.get("rows", []):
                return
            table = PrettyTable()
            table.title = sheet_name
            table.field_names = data.get("headers", [])
            for row in data.get("rows", []):
                table.add_row(row)
            print(table)
