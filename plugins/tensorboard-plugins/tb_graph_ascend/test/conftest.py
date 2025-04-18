import pytest
import json
import os
from pathlib import Path


def load_st_test_cases():
    meta_path = Path(__file__).parent / "data/metadata_st.json"  
    with open(meta_path) as f:
        return json.load(f)["test_cases"]  

    
def load_ut_test_cases():
    meta_path = Path(__file__).parent / "data/metadata_ut.json" 
    with open(meta_path) as f:
        return json.load(f)["test_cases"]  


# 动态生成测试用例
def pytest_generate_tests(metafunc):
    if "meta_data" in metafunc.fixturenames and "operation" in metafunc.fixturenames:
        ut_test_cases = load_st_test_cases()
        params = []
        for case in ut_test_cases:
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
    if "ut_test_case" in metafunc.fixturenames:
        ut_test_cases = load_ut_test_cases()
        params = []
        for case in ut_test_cases:
            params.append(pytest.param(
                case,
                id=f"{case['type']}-{case['name']}"
            ))
        # 确保参数名称与参数值数量一致
        metafunc.parametrize("ut_test_case", params)


@pytest.fixture
def meta_data(meta_data):
    # 返回当前测试的操作配置
    return meta_data


@pytest.fixture
def operation_config(operation):
    # 返回当前测试的操作配置
    return operation


@pytest.fixture
def ut_test_case(ut_test_case):
    # 返回当前测试的操作配置
    return ut_test_case
