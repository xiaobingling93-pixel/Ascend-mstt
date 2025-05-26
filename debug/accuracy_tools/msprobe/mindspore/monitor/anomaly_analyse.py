# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
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

import os

from msprobe.core.common.log import logger
from msprobe.core.common.const import MonitorConst
from msprobe.core.common.file_utils import save_json, create_directory, remove_path, \
    check_file_or_directory_path, load_json


class AnomalyDataWriter:
    """
    异常数据写入类，负责将异常数据写入到JSON文件中。
    """

    def __init__(self, dump_path, rank) -> None:
        self.dump_path = dump_path
        self.dump_rank_dir = os.path.join(self.dump_path, f"rank{rank}")
        self.json_path = os.path.join(self.dump_rank_dir, MonitorConst.ANOMALY_JSON)

    @staticmethod
    def get_anomaly_dict(anomalies):
        """将GradAnomalyData列表转换为json"""
        anomalies_json = {}
        for anomaly in anomalies:
            anomalies_json.update({anomaly.get_key(): anomaly.to_dict()})
        return anomalies_json

    def init_detected_json(self):
        """初始化落盘文件"""
        create_directory(self.dump_rank_dir)

        if os.path.exists(self.json_path):
            check_file_or_directory_path(self.json_path, isdir=False)
            logger.warning(f"The existing file will be deleted: {self.json_path}.")
            remove_path(self.json_path)
        save_json(self.json_path, {}, indent=1)

    def write_detected_json(self, anomalies):
        """
        落盘异常数据
        Args:
        anomalies: GradAnomalyData对象列表
        """
        anomalies_json = self.get_anomaly_dict(anomalies)
        logger.info(f"{MonitorConst.ANOMALY_JSON} is at {self.dump_rank_dir}.")

        data_to_write = load_json(self.json_path) if os.path.exists(self.json_path) else {}
        data_to_write.update(anomalies_json)
        save_json(self.json_path, data_to_write, indent=1)
