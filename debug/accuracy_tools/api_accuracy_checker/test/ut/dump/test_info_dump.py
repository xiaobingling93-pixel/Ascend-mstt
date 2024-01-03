import unittest
import os
from unittest.mock import patch
from api_accuracy_checker.dump.api_info import APIInfo, BackwardAPIInfo
from api_accuracy_checker.dump.info_dump import write_api_info_json


class TestInfoDump(unittest.TestCase):

    def test_write_api_info_json_backward(self):
        api_info = BackwardAPIInfo("test_backward_api", [1, 2, 3])
        with patch('api_accuracy_checker.dump.info_dump.write_json') as mock_write_json:
            write_api_info_json(api_info)
            rank = os.getpid()
            mock_write_json.assert_called_with(f'./step2/backward_info_{rank}.json', api_info.grad_info_struct)

    def test_write_api_info_json_invalid_type(self):
        api_info = APIInfo("test_api", APIInfo.get_full_save_path("save_path", "forward_real_data", contain_step=True),
                           is_save_data=True)
        with self.assertRaises(ValueError):
            write_api_info_json(api_info)

    def tearDown(self):
        rank = os.getpid()
        files = [f'./step2/backward_info_{rank}.json']
        for file in files:
            if os.path.exists(file):
                os.remove(file)