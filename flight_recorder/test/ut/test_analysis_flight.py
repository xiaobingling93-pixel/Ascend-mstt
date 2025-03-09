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

import os
import shutil
import pickle

from unittest.mock import patch

import pytest

from flight_recorder.flight_recorder_analyze.analysis_flight import (
    load_recorder_data,
    extract_hccl_info,
    analyze_pg_groups,
    main,
    SafeUnpickler,
)


WORLD_SIZE = 2
TEST_FLIGHT_RECORDER_PATH = "./test_fight_recorder_file"


class UmaskWrapper:
    """Write with preset umask
    >>> with UmaskWrapper():
    >>>     ...
    """

    def __init__(self, umask=0o027):
        self.umask, self.ori_umask = umask, None

    def __enter__(self):
        self.ori_umask = os.umask(self.umask)

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        os.umask(self.ori_umask)


TEST_RECORDER_DATA = {
    "entries": [
        {
            "state": "scheduled",
            "record_id": 1,
            "pg_id": "pg1",
            "time_discovered_completed_ns": 1000,
            "frames": [{"name": "op1"}],
        },
        {
            "state": "completed",
            "record_id": 2,
            "pg_id": "pg1",
            "time_discovered_completed_ns": 2000,
            "frames": [{"name": "op2"}],
        },
    ]
}


@pytest.fixture
def temp_dir():
    """创建一个临时目录，并在其中生成模拟的 recorder 数据文件。"""
    with UmaskWrapper():
        os.mkdir(TEST_FLIGHT_RECORDER_PATH)
    for rank in range(WORLD_SIZE):
        file_path = os.path.join(TEST_FLIGHT_RECORDER_PATH, str(rank))
        with UmaskWrapper():
            with open(file_path, "wb") as f:
                pickle.dump(TEST_RECORDER_DATA, f)
    yield TEST_FLIGHT_RECORDER_PATH

    if os.path.exists(TEST_FLIGHT_RECORDER_PATH):
        shutil.rmtree(TEST_FLIGHT_RECORDER_PATH)


def test_main(temp_dir):
    with patch("sys.argv", ["analysis_flight.py", TEST_FLIGHT_RECORDER_PATH, "2"]):
        main()


def test_load_recorder_data(temp_dir):
    """测试 load_recorder_data 函数是否正确加载 recorder 数据。"""
    recorder_dict = load_recorder_data(TEST_FLIGHT_RECORDER_PATH, WORLD_SIZE)
    assert len(recorder_dict) == WORLD_SIZE


def test_extract_hccl_info():
    """测试 extract_hccl_info 函数是否正确提取 HCCL 信息。"""
    recorder_dict = {str(rank): TEST_RECORDER_DATA for rank in range(WORLD_SIZE)}
    hccl_dict = extract_hccl_info(recorder_dict)
    assert len(hccl_dict) == WORLD_SIZE
    for _, info in hccl_dict.items():
        assert info["state"] == "completed"
        assert info["record_id"] == 2
        assert info["pg_id"] == "pg1"
        assert info["time_discovered_completed_ns"] == 2000
        assert info["name"] == "op2"


def test_analyze_pg_groups():
    hccl_dict_list = [
        {
            "0": {
                "state": "scheduled",
                "record_id": 1,
                "pg_id": "pg1",
                "time_discovered_completed_ns": 1000,
                "name": "op1",
            },
            "1": {
                "state": "scheduled",
                "record_id": 1,
                "pg_id": "pg1",
                "time_discovered_completed_ns": 1000,
                "name": "op1",
            },
        },
        {
            "0": {
                "state": "completed",
                "record_id": 1,
                "pg_id": "pg1",
                "time_discovered_completed_ns": 2000,
                "name": "op2",
            },
            "1": {
                "state": "completed",
                "record_id": 1,
                "pg_id": "pg1",
                "time_discovered_completed_ns": 2000,
                "name": "op2",
            },
        },
        {
            "0": {
                "state": "scheduled",
                "record_id": 1,
                "pg_id": "pg1",
                "time_discovered_completed_ns": 2000,
                "name": "op2",
            },
            "1": {
                "state": "completed",
                "record_id": 1,
                "pg_id": "pg1",
                "time_discovered_completed_ns": 2000,
                "name": "op2",
            },
        },
    ]
    for data in hccl_dict_list:
        analyze_pg_groups(data)
