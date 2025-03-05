# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import logging

COMMAND_SUCCESS = 0


def execute_cmd(cmd):
    logging.info('Execute command:%s' % " ".join(cmd))
    completed_process = subprocess.run(cmd, shell=False, stderr=subprocess.PIPE)
    if completed_process.returncode != COMMAND_SUCCESS:
        logging.error(completed_process.stderr.decode())
    return completed_process.returncode


def check_column_actual(actual_columns, expected_columns, context):
    """检查实际列名是否与预期列名一致"""
    missing = set(expected_columns) - set(actual_columns)  # O(n + m)
    for col in missing:
        logging.error(f"在 {context} 中未找到预期列名: {col}")
    return len(missing) == 0


def check_row(df, expected_columns, numeric_columns):
    """检查数据框中Metric列数据类型和指定列数据是否为数字"""
    # 检查Metric列的数据类型是否为字符串
    for row_index in df.index:
        try:
            value = df.at[row_index, 'Metric']
            if not isinstance(value, str):
                logging.error(f"在Metric列的第{row_index}行，值 '{value}' 不是字符串类型")
                return False
        except KeyError:
            logging.error(f"数据框中不存在 'Metric' 列")
            return False

    # 检查其他列的数据是否为数字
    for column in numeric_columns:
        if column not in df.columns:
            logging.error(f"数据框中不存在 {column} 列")
            continue
        for row_index in df.index:
            try:
                cell_value = df.at[row_index, column]
                float(cell_value)
            except (ValueError, KeyError):
                logging.error(
                    f"在 {column} 列的第 {row_index} 行，值 {cell_value} 不是有效的数字")
                return False
    return True