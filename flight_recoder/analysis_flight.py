# Copyright (c) 2025, Huawei Technologies Co., Ltd.
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

# Copyright Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

import os
import pickle
import sys
import logging
from collections import defaultdict

from check_path import get_valid_read_path


logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # 设置日志格式
    handlers=[logging.StreamHandler()],  # 输出到控制台
)


SAFE_CLASSES = {
    # 内置安全类型
    "builtins": {"str", "int", "float", "list", "dict", "tuple"},
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 检查模块和类是否在白名单中
        if module in SAFE_CLASSES and name in SAFE_CLASSES[module]:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Forbidden class: {module}.{name}")


def load_recorder_data(path, world_size):
    """加载所有 rank 的 recorder 数据"""
    recorder_dict = {}
    for rank in range(world_size):
        file_path = os.path.join(path, str(rank)) if not path.endswith("/") else path + str(rank)
        file_path = get_valid_read_path(file_path)
        try:
            with open(file_path, "rb") as f:
                res = SafeUnpickler(f).load()
                recorder_dict[str(rank)] = res
        except Exception as e:
            logging.error(f"Failed to load data from {file_path}: {e}")
    return recorder_dict


def extract_hccl_info(recorder_dict):
    """从 recorder 数据中提取 HCCL 相关信息"""
    hccl_dict = {}
    for rank, recorder in recorder_dict.items():
        entries = recorder.get("entries", [])
        if not entries:
            continue
        last_entry = entries[-1]
        hccl_dict[rank] = {
            "state": last_entry.get("state", None),
            "record_id": last_entry.get("record_id", None),
            "pg_id": last_entry.get("pg_id", None),
            "time_discovered_completed_ns": last_entry.get("time_discovered_completed_ns", None),
            "name": last_entry.get("frames", [{}])[0].get("name", None),
        }
    return hccl_dict


def analyze_pg_groups(hccl_dict):
    """分析 HCCL 数据，按 pg_id 分组并检查问题"""
    pg_groups = defaultdict(list)
    for _, op in hccl_dict.items():
        pg_groups[op["pg_id"]].append(op)

    for pg_id, group in pg_groups.items():
        scheduled_ops = [op for op in group if op["state"] == "scheduled"]
        completed_ops = [op for op in group if op["state"] == "completed"]

        # 情况 1: 所有卡都是 scheduled，且 record_id 和 name 相同
        if len(scheduled_ops) == len(group):
            record_id = scheduled_ops[0]["record_id"]
            name = scheduled_ops[0]["name"]
            all_same = all(op["record_id"] == record_id and op["name"] == name for op in scheduled_ops)
            if all_same:
                logging.info(
                    f"The pg_id {pg_id}'s Communication Operator {name}"
                    " executed too slowly, causing the HCCL to time out."
                )

        # 情况 2: 存在 completed 算子且 该算子的record_id 比其他 scheduled 算子少 1
        elif completed_ops and scheduled_ops:
            completed_op = completed_ops[0]
            scheduled_record_id = scheduled_ops[0]["record_id"]
            if completed_op["record_id"] == scheduled_record_id - 1:
                logging.info(
                    f"The pg_id {pg_id}'s rank {completed_op['pg_id']}'s "
                    "Computational task took too long, causing the other ranks' "
                    "HCCL task to time out."
                )

        # 情况 3: 所有算子均为 completed
        elif not scheduled_ops and completed_ops:
            latest_op = max(completed_ops, key=lambda x: x["time_discovered_completed_ns"] or 0)
            logging.info(
                f"The computational task of the pg_id {pg_id} "
                f"after the communication operator {latest_op['name']} " 
                "took too long."
            )

        else:
            logging.info(f"The situation cannot be recognized!")


def get_int_arg(args, idx, default):
    if len(args) > idx:
        try:
            return int(args[idx])
        except ValueError:
            logging.warning(f"Invalid input {args[idx]}, using default: {default}")
    return default


def main():
    # 设置默认值
    default_path = os.getenv("TORCH_HCCL_DEBUG_INFO_TEMP_FILE")
    default_world_size = 8

    # 获取命令行参数，如果未提供则使用默认值
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    world_size = get_int_arg(sys.argv, 2, default_world_size)

    if not path:
        raise ValueError("Path is required and cannot be empty.")

    logging.info(f"Path: {path}")
    logging.info(f"World Size: {world_size}")

    # 加载数据
    recorder_dict = load_recorder_data(path, world_size)
    if not recorder_dict:
        logging.error("No valid recorder data found.")
        return

    # 提取 HCCL 信息
    hccl_dict = extract_hccl_info(recorder_dict)

    # 分析 HCCL 数据
    analyze_pg_groups(hccl_dict)


if __name__ == "__main__":
    main()