import unittest
from unittest.mock import patch, MagicMock
from msprobe.mindspore.api_accuracy_checker.api_accuracy_checker import api_checker_main


class TestApiCheckerMain(unittest.TestCase):

    @patch('msprobe.mindspore.api_accuracy_checker.api_accuracy_checker.ApiAccuracyChecker')
    def test_api_checker_main(self, MockApiAccuracyChecker):
        # 创建 Mock 实例
        mock_instance = MockApiAccuracyChecker.return_value

        # 设置 Mock 方法的返回值
        mock_instance.parse = MagicMock()
        mock_instance.run_and_compare = MagicMock()
        mock_instance.to_detail_csv = MagicMock()
        mock_instance.to_result_csv = MagicMock()

        # 模拟输入参数
        class Args:
            api_info_file = "test.json"
            out_path = "./"

        args = Args()

        # 调用要测试的函数
        api_checker_main(args)

        # 验证各个方法是否被调用
        mock_instance.parse.assert_called_once_with(args.api_info_file)
        mock_instance.run_and_compare.assert_called_once()
        mock_instance.to_detail_csv.assert_called_once_with(args.out_path)
        mock_instance.to_result_csv.assert_called_once_with(args.out_path)


if __name__ == "__main__":
    unittest.main()
