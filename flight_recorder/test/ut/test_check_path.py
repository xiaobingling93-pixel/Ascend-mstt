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
import stat
import json

import yaml
import pytest

from flight_recorder.flight_recorder_analyze.check_path import (
    get_valid_path,
    get_valid_read_path,
    check_type,
    type_to_str,
)

TEST_DIR = "/tmp/a_test_path_for_testing_check_path_common/"
TEST_READ_FILE_NAME = TEST_DIR + "testfile.testfile"
USER_NOT_PERMITTED_READ_FILE = TEST_DIR + "testfile_not_readable.testfile"
OTHERS_READABLE_READ_FILE = TEST_DIR + "testfile_others_readable.testfile"
OTHERS_WRITABLE_READ_FILE = TEST_DIR + "testfile_others_writable.testfile"
USER_NOT_PERMITTED_WRITE_FILE = TEST_DIR + "testfile_not_writable/foo"
JSON_FILE = TEST_DIR + "testfile.json"
YAML_FILE = TEST_DIR + "testfile.yaml"
TEST_FILE = TEST_DIR + "testfile.test"
ORI_DATA = {"a_long_key_name": 1, 12: "b", 3.14: "", "c": {"d": 3, "e": 4}, True: "true", False: "false", None: "null"}
OVER_WRITE_DATA = {"hello": "world"}


def setup_module():
    os.makedirs(TEST_DIR, mode=int("700", 8), exist_ok=True)

    default_mode = stat.S_IWUSR | stat.S_IRUSR  # 600
    with os.fdopen(
        os.open(TEST_READ_FILE_NAME, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=default_mode), "w"
    ) as temp_file:
        temp_file.write("a_test_file_name_for_testing_automl_common")

    with os.fdopen(os.open(USER_NOT_PERMITTED_READ_FILE, os.O_CREAT, mode=000), "w"):
        pass

    with os.fdopen(os.open(OTHERS_READABLE_READ_FILE, os.O_CREAT, mode=default_mode), "w"):
        pass
    os.chmod(OTHERS_READABLE_READ_FILE, int("755", 8))

    with os.fdopen(os.open(OTHERS_WRITABLE_READ_FILE, os.O_CREAT, mode=default_mode), "w"):
        pass
    os.chmod(OTHERS_WRITABLE_READ_FILE, int("666", 8))

    dir_name = os.path.dirname(USER_NOT_PERMITTED_WRITE_FILE)
    os.makedirs(dir_name, mode=int("500", 8), exist_ok=True)

    with os.fdopen(os.open(JSON_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=default_mode), "w") as json_file:
        json.dump(ORI_DATA, json_file)

    with os.fdopen(os.open(YAML_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=default_mode), "w") as yaml_file:
        yaml.dump(ORI_DATA, yaml_file)


def teardown_module():
    os.remove(TEST_READ_FILE_NAME)
    os.chmod(USER_NOT_PERMITTED_READ_FILE, int("600", 8))
    os.remove(USER_NOT_PERMITTED_READ_FILE)
    os.remove(OTHERS_READABLE_READ_FILE)
    os.remove(OTHERS_WRITABLE_READ_FILE)

    dir_name = os.path.dirname(USER_NOT_PERMITTED_WRITE_FILE)
    os.chmod(dir_name, int("700", 8))
    os.removedirs(dir_name)

    os.remove(JSON_FILE)
    os.remove(YAML_FILE)
    if os.path.exists(TEST_FILE):
        os.remove(TEST_FILE)

    os.removedirs(TEST_DIR)


def test_check_type_given_valid_when_any_then_pass():
    check_type(12, value_type=int)


def test_check_type_given_int_when_str_then_error():
    with pytest.raises(TypeError):
        # TypeError: test must be str, not int.
        check_type(12, value_type=str, param_name="test")


def test_get_valid_path_given_valid_when_any_then_pass():
    get_valid_path("../anypath")
    get_valid_path("../anypath/a")


def test_get_valid_path_given_invalid_when_any_then_value_error():
    with pytest.raises(ValueError):
        get_valid_path("../anypath*a")  # ValueError: ../anypath*a contains invalid characters.
    with pytest.raises(ValueError):
        get_valid_path("../anypath/\\a")  # ValueError: ../anypath/\a contains invalid characters.
    with pytest.raises(ValueError):
        get_valid_path("../anypath/!a")  # ValueError: ../anypath/!a contains invalid characters.


def test_get_valid_read_path_given_valid_when_any_then_pass():
    get_valid_read_path(TEST_READ_FILE_NAME)
    get_valid_read_path(OTHERS_READABLE_READ_FILE)
    get_valid_read_path(OTHERS_WRITABLE_READ_FILE, check_user_stat=False)


def test_get_valid_read_path_given_invalid_when_any_then_value_error():
    with pytest.raises(ValueError):
        get_valid_read_path("./not_exist")  # ValueError: The file ... doesn't exist or not a file.
    with pytest.raises(ValueError):
        # ValueError: The file ... exceeds size limitation of 1.
        get_valid_read_path(TEST_READ_FILE_NAME, size_max=1)
    with pytest.raises(ValueError):
        # ValueError: Current user doesn't have read permission to the file ....
        get_valid_read_path(USER_NOT_PERMITTED_READ_FILE)
    with pytest.raises(ValueError):
        # ValueError: The file ... has others writable permission.
        get_valid_read_path(OTHERS_WRITABLE_READ_FILE)
