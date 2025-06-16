import pytest
from server.app.utils.global_state import GraphState
from tests.data.test_case_factory import TestCaseFactory


@pytest.fixture(scope="function", autouse=True)
def reset_global_state():
    """每个测试后重置全局状态"""
    
    # 执行测试
    yield
    
    # 恢复原始状态
    GraphState.init_defaults()


def pytest_addoption(parser):
    """添加自定义命令行选项"""
    parser.addoption("--runslow", action="store_true", default=False,
                    help="Run slow tests")
    parser.addoption("--dataset", action="store", default="small",
                    help="Test dataset size: small|medium|large")


@pytest.fixture
def test_case_factory():
    """提供测试用例工厂实例"""
    return TestCaseFactory
