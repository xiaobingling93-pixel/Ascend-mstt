import pytest
import json
import os
from pathlib import Path


def load_st_test_cases():
    meta_path = Path(__file__).parent / "data/metadata_st.json"  # 修改文件扩展名为 .json
    with open(meta_path) as f:
        return json.load(f)["test_cases"]  # 使用 json.load 代替 yaml.safe_load


# 动态生成测试用例
def pytest_generate_tests(metafunc):
    if "meta_data" in metafunc.fixturenames and "operation" in metafunc.fixturenames:
        test_cases = load_st_test_cases()
        params = []
        for case in test_cases:
            metaData = case["meta_data"]
            metaData["run"] = Path(__file__).parent / metaData["run"]
            for op in case["operations"]:
                params.append(pytest.param(
                    metaData,
                    op,
                    id=f"{metaData['run']}-{op['type']}"
                ))
        # 确保参数名称与参数值数量一致
        metafunc.parametrize("meta_data, operation", params)


@pytest.fixture
def graph_data(meta_data):
    # 返回当前测试的操作配置
    return meta_data


@pytest.fixture
def operation_config(operation):
    # 返回当前测试的操作配置
    return operation
