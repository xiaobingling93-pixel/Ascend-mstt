# coding=utf-8
import unittest
import time
from datetime import datetime, timezone
from ptdbg_ascend.common import utils as common
#from ptdbg_ascend.common import CompareException

class TestCommonUtilsMethods(unittest.TestCase):

    def test_VersionCheck(self):
        V0_1 = "0.1"
        V1_8 = "1.8"
        V1_11 = "1.11"
        V2_0 = "2.0"
        V2_1 = "2.1"
        V2_2 = "2.2"
        version_check = common.VersionCheck
        self.assertFalse(version_check.check_torch_version(V0_1))
        self.assertTrue(version_check.check_torch_version(V1_8) or version_check.check_torch_version(V1_11) or 
                        version_check.check_torch_version(V2_0) or version_check.check_torch_version(V2_1)
                        or version_check.check_torch_version(V2_2))

    def test_check_mode_valid(self):
        mode_check = common.check_mode_valid
        self.assertEqual(mode_check("all"), None)
        self.assertEqual(mode_check("list",scope=["Tensor_permute_1_forward", "Tensor_transpose_2_forward", "Torch_relu_3_backward"]), None)
        self.assertEqual(mode_check("range", scope=["Tensor_abs_1_forward", "Tensor_transpose_3_forward"]), None)
        self.assertEqual(mode_check("stack",scope=["Tensor_abs_1_forward", "Tensor_transpose_3_forward"]), None)
        self.assertEqual(mode_check("acl",scope=["Tensor_permute_1_forward"]), None)
        self.assertEqual(mode_check("api_list",api_list=["relu"]), None)
        self.assertEqual(mode_check("api_stack"), None)
        self.assertRaises(common.CompareException, mode_check, "api_stack_123")

    def test_parse_value_by_comma(self):
        data = [1, 2, 4, 8]
        self.assertEqual(common.parse_value_by_comma("1,2,4,8"), data)

    def test_get_data_len_by_shape(self):
        getshape = common.get_data_len_by_shape
        data = [1, 2, 4, 8]
        self.assertEqual(getshape(data), 64)
        data = [-1, 2, 4, 8]
        self.assertEqual(getshape(data), -1)

    def test_add_time_as_suffix(self):
        name = "op_cmp"
        csv_name = '{}_{}.csv'.format(name, time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))
        self.assertEqual(common.add_time_as_suffix(name), csv_name)

    def test_add_time_with_xlsx(self):
        name = "op_cmp"
        xlsx_name = '{}_{}.xlsx'.format(name, time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))
        self.assertEqual(common.add_time_with_xlsx(name), xlsx_name)

    def test_get_time(self):
        time = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.assertEqual(common.get_time(), time)

    def test_format_value(self):
        value = 12345.6789
        format_value = float('{:.12f}'.format(value))
        self.assertEqual(common.format_value(value), format_value)

    def test_modify_dump_path(self):
        dump_path = "/usr/dump"
        mode = "api_stack"
        self.assertEqual(common.modify_dump_path(dump_path, mode), "/usr/api_stack_dump")

    def test_create_directory(self):
        pass

    def test_execute_command(self):
        pass

    def test_save_numpy_data(self):
        pass

    def test_torch_device_guard(self):
        pass

    def test_seed_all(self):
        pass

    def test_get_process_rank(self):
        pass

    def test_check_file_size(self):
        pass

    def test_get_dump_data_path(self):
        pass