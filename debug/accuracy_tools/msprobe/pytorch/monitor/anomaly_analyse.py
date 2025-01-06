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
import sys
import argparse
import ast
import heapq

from msprobe.pytorch.common.log import logger
from msprobe.core.common.const import MonitorConst
from msprobe.core.common.file_utils import check_path_before_create, save_json, create_directory, remove_path, \
    check_file_or_directory_path, load_json
from msprobe.pytorch.monitor.anomaly_detect import GradAnomalyData


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
        check_path_before_create(self.dump_path)
        if not os.path.exists(self.dump_path):
            create_directory(self.dump_path)

        if not os.path.exists(self.dump_rank_dir):
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


class AnomalyDataLoader:
    def __init__(self, data_path) -> None:
        self.data_path = data_path

    @staticmethod
    def create_instances_from_dict(anomalies_dict: dict):
        instances = []
        for values in anomalies_dict.values():
            try:
                instances.append(GradAnomalyData(**values))
            except KeyError as e:
                logger.warning(f"Missing key in anomaly data: {e}.")
            except Exception as e:
                logger.warning(f"Value error when creating a GradAnomalyData instance: {e}.")
        return instances

    def get_anomalies_from_jsons(self):
        """遍历文件夹,从rankK/anomaly.json中读取异常数据
        return: anomalies: GradAnomalyData对象列表
        """
        anomalies = []
        check_file_or_directory_path(self.data_path, isdir=True)
        for rank_dir in os.listdir(self.data_path):
            rank_path = os.path.join(self.data_path, rank_dir)
            if not os.path.isdir(rank_path):
                continue
            json_path = os.path.join(rank_path, MonitorConst.ANOMALY_JSON)
            if not os.path.exists(json_path):
                continue
            data_anomalies = load_json(json_path)
            instances = self.create_instances_from_dict(data_anomalies)
            anomalies.extend(instances)
        return anomalies


class AnomalyAnalyse:
    def __init__(self) -> None:
        self.sorted_anomalies = []

    def get_range_top_k(self, topk, step_list, anomalies):
        """
        获取前topk个step_list范围内的异常。
        """
        if not step_list:
            filtered_anomalies = anomalies
        else:
            filtered_anomalies = [
                anomaly
                for anomaly in anomalies
                if anomaly.step in step_list
            ]
        if topk >= len(filtered_anomalies):
            self.sorted_anomalies = sorted(filtered_anomalies)
        else:
            self.sorted_anomalies = list(heapq.nsmallest(topk, filtered_anomalies))
        return self.sorted_anomalies

    def rewrite_sorted_anomalies(self, output_path):
        """
        将排序后的异常数据重新落盘
        """
        check_file_or_directory_path(output_path, isdir=True)

        sorted_data = AnomalyDataWriter.get_anomaly_dict(self.sorted_anomalies)
        logger.info(f"{MonitorConst.ANALYSE_JSON} is at {output_path}.")
        json_path = os.path.join(output_path, MonitorConst.ANALYSE_JSON)
        if os.path.exists(json_path):
            logger.warning(f"The existing file will be deleted: {json_path}.")
            remove_path(json_path)
        save_json(json_path, sorted_data, indent=1)


def _get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", dest="data_path_dir", default="./", type=str,
                        help="<Required> The anomaly detect result dictionary: generate from monitor tool.",
                        required=True,
                        )
    parser.add_argument("-o", "--out_path", dest="out_path", default="", type=str,
                        help="<optional> The analyse task result out path.",
                        required=False,
                        )
    parser.add_argument("-k", "--topk", dest="top_k_number", default=8, type=int,
                        help="<optional> Top K number of earliest anomalies.",
                        required=False,
                        )
    parser.add_argument("-s", "--step", dest="step_list", default="[]", type=str,
                        help="<optional> Analyse which steps.",
                        required=False,
                        )
    return parser.parse_args(sys.argv[1:])


def _get_step_and_stop(args):
    try:
        step_list = ast.literal_eval(args.step_list)
        if not isinstance(step_list, list):
            raise ValueError(f"{args.step_list} is not a list.")
    except (ValueError, SyntaxError, RecursionError) as e:
        raise Exception(f"The step list must be a resolvable list type.") from e
    if args.top_k_number <= 0:
        raise Exception("The top k number must be greater than 0.")
    return step_list, args.top_k_number


def _anomaly_analyse():
    args = _get_parse_args()
    step_list, top_k_number = _get_step_and_stop(args)
    loader = AnomalyDataLoader(args.data_path_dir)
    anomalies = loader.get_anomalies_from_jsons()
    analyser = AnomalyAnalyse()
    top_anomalies = analyser.get_range_top_k(
        top_k_number, step_list, anomalies
    )
    analyser.rewrite_sorted_anomalies(
        args.out_path if args.out_path else args.data_path_dir
    )

    logger.info(f"Top {top_k_number} anomalies are listed as follows:")
    for index, anomaly in enumerate(top_anomalies):
        logger.info(f"{index}: {anomaly.message}")


if __name__ == "__main__":
    _anomaly_analyse()
    logger.info("Analyse task completed.")
