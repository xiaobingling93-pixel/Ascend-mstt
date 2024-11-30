import os
import unittest
from unittest.mock import patch, MagicMock

from msprobe.pytorch.monitor.anomaly_analyse import AnomalyDataWriter, AnomalyDataLoader, AnomalyAnalyse, \
    _get_parse_args, _get_step_and_stop, _anomaly_analyse


class TestAnomalyDataWriter(unittest.TestCase):

    def test_get_anomaly_dict(self):
        # 测试 get_anomaly_dict 方法
        anomaly1 = MagicMock()
        anomaly1.get_key.return_value = 'anomaly1'
        anomaly1.to_dict.return_value = {'value': 1}

        anomaly2 = MagicMock()
        anomaly2.get_key.return_value = 'anomaly2'
        anomaly2.to_dict.return_value = {'value': 2}

        anomalies = [anomaly1, anomaly2]
        result = AnomalyDataWriter.get_anomaly_dict(anomalies)

        expected = {
            'anomaly1': {'value': 1},
            'anomaly2': {'value': 2}
        }
        self.assertEqual(result, expected)

    @patch('msprobe.pytorch.monitor.anomaly_analyse.os.path.exists')
    @patch('msprobe.pytorch.monitor.anomaly_analyse.create_directory')
    @patch('msprobe.pytorch.monitor.anomaly_analyse.save_json')
    def test_init_detected_json(self, mock_save_json, mock_create_directory, mock_exists):
        # 模拟路径检查
        mock_exists.side_effect = [False, False, False]  # dump_path, dump_rank_dir, json_path
        # 模拟文件不存在
        mock_create_directory.side_effect = None

        writer = AnomalyDataWriter('/tmp/dump', 0)
        writer.init_detected_json()

        # 检查是否创建了目录
        mock_create_directory.assert_any_call('/tmp/dump')
        mock_create_directory.assert_any_call('/tmp/dump/rank0')

        # 检查是否初始化了 JSON 文件
        mock_save_json.assert_called_once_with(writer.json_path, {}, indent=1)

    @patch('msprobe.pytorch.monitor.anomaly_analyse.check_file_or_directory_path')
    @patch('msprobe.pytorch.monitor.anomaly_analyse.remove_path')
    @patch('msprobe.pytorch.monitor.anomaly_analyse.save_json')
    @patch('msprobe.pytorch.monitor.anomaly_analyse.logger')
    def test_init_detected_json_existing_file(self, mock_logger, mock_save_json, mock_remove_path, mock_check_path):
        # 设置测试参数
        dump_path = 'test/dump_path'
        rank = 0
        writer = AnomalyDataWriter(dump_path, rank)

        # 模拟文件存在情况
        mock_check_path.side_effect = None  # 阻止实际调用
        mock_remove_path.return_value = None  # 阻止实际调用

        # 模拟 json_path 存在
        writer.json_path = 'existing_file.json'
        with patch('os.path.exists', return_value=True):
            writer.init_detected_json()

        # 验证文件删除和新文件保存
        mock_remove_path.assert_called_once_with(writer.json_path)
        mock_logger.warning.assert_called_once_with(f"The existing file will be deleted: {writer.json_path}.")
        mock_save_json.assert_called_once_with(writer.json_path, {}, indent=1)

    @patch('msprobe.pytorch.monitor.anomaly_analyse.os.path.exists')
    @patch('msprobe.pytorch.monitor.anomaly_analyse.load_json')
    @patch('msprobe.pytorch.monitor.anomaly_analyse.save_json')
    def test_write_detected_json(self, mock_save_json, mock_load_json, mock_exists):
        mock_exists.side_effect = [True, True]  # json_path 存在

        # 创建模拟的异常数据
        anomalies = [MagicMock(), MagicMock()]
        anomalies[0].get_key.return_value = 'anomaly1'
        anomalies[0].to_dict.return_value = {'value': 1}
        anomalies[1].get_key.return_value = 'anomaly2'
        anomalies[1].to_dict.return_value = {'value': 2}

        mock_load_json.return_value = {'existing_anomaly': {'value': 0}}

        writer = AnomalyDataWriter('/tmp/dump', 0)
        writer.write_detected_json(anomalies)

        expected_data = {
            'existing_anomaly': {'value': 0},
            'anomaly1': {'value': 1},
            'anomaly2': {'value': 2}
        }

        # 检查 JSON 是否被加载和保存
        mock_load_json.assert_called_once_with(writer.json_path)
        mock_save_json.assert_called_once_with(writer.json_path, expected_data, indent=1)


class TestAnomalyDataLoader(unittest.TestCase):

    @patch('msprobe.pytorch.monitor.anomaly_analyse.GradAnomalyData')  # 替换为 GradAnomalyData 的实际导入路径
    def test_create_instances_from_dict(self, mock_GradAnomalyData):
        # 模拟 GradAnomalyData 的构造函数
        def mock_constructor(**kwargs):
            return None
        mock_GradAnomalyData.side_effect = mock_constructor  # 假设构造成功

        data = {
            'anomaly1': {'key1': 'value1', 'key2': 'value2'},
            'anomaly2': {'key1': 'value3', 'key2': 'value4'},
        }

        loader = AnomalyDataLoader('/tmp/data')
        instances = loader.create_instances_from_dict(data)

        # 确保创建了两个实例，第三个因缺少 key2 被捕获
        self.assertEqual(len(instances), 2)

    @patch('msprobe.pytorch.monitor.anomaly_analyse.os.listdir')
    @patch('msprobe.pytorch.monitor.anomaly_analyse.os.path.exists')
    @patch('msprobe.pytorch.monitor.anomaly_analyse.load_json')
    @patch('msprobe.pytorch.monitor.anomaly_analyse.check_file_or_directory_path')
    @patch('msprobe.pytorch.monitor.anomaly_analyse.GradAnomalyData')
    def test_get_anomalies_from_jsons(self, mock_GradAnomalyData, mock_check_path, mock_load_json, \
                                      mock_exists, mock_listdir):
        mock_check_path.return_value = None
        mock_listdir.return_value = ['rank0', 'rank1']
        
        # 模拟 rank0/anomaly.json 存在，rank1/anomaly.json 不存在
        mock_exists.side_effect = [True, False]
        mock_load_json.return_value = {
            'anomaly1': {'key1': 'value1', 'key2': 'value2'},
            'anomaly2': {'key1': 'value3', 'key2': 'value4'}
        }

        # 模拟 GradAnomalyData 的构造函数
        def mock_constructor(**kwargs):
            return None
        mock_GradAnomalyData.side_effect = mock_constructor  # 假设构造成功

        loader = AnomalyDataLoader('/tmp/data')
        with patch('msprobe.pytorch.monitor.anomaly_analyse.os.path.isdir', return_value = True):
            anomalies = loader.get_anomalies_from_jsons()

        # 确保从 rank0 读取了异常数据
        self.assertEqual(len(anomalies), 2)
        mock_check_path.assert_called_once_with('/tmp/data', isdir=True)
        mock_load_json.assert_called_once_with('/tmp/data/rank0/anomaly.json')


class TestAnomalyAnalyse(unittest.TestCase):

    def setUp(self):
        self.anomaly_analyse = AnomalyAnalyse()
        self.anomalies = [
            MagicMock(step=1, value=5),
            MagicMock(step=2, value=3),
            MagicMock(step=3, value=8),
            MagicMock(step=4, value=1),
        ]

    @patch('msprobe.pytorch.monitor.anomaly_analyse.heapq.nsmallest')
    def test_get_range_top_k(self, mock_nsmallest):
        # 设置 mock 的返回值
        mock_nsmallest.return_value = self.anomalies[:2]

        # 测试 step_list 为空
        result = self.anomaly_analyse.get_range_top_k(2, [], self.anomalies)
        self.assertEqual(result, [self.anomalies[0], self.anomalies[1]])

        # 测试 step_list 不为空
        step_list = [1, 2, 3]
        result = self.anomaly_analyse.get_range_top_k(2, step_list, self.anomalies)
        self.assertEqual(result, [self.anomalies[0], self.anomalies[1]])  # 应该是 value=3 和 value=5 的异常

    @patch('msprobe.pytorch.monitor.anomaly_analyse.os.path.exists')
    @patch('msprobe.pytorch.monitor.anomaly_analyse.AnomalyDataWriter.get_anomaly_dict')
    @patch('msprobe.pytorch.monitor.anomaly_analyse.save_json')
    @patch('msprobe.pytorch.monitor.anomaly_analyse.logger')
    def test_rewrite_sorted_anomalies(self, mock_logger, mock_save_json, mock_get_anomaly_dict, mock_exists):
        # 设置 mock
        mock_exists.return_value = False
        mock_get_anomaly_dict.return_value = {'anomalies': 'data'}

        output_path = 'output_path'

        # 调用方法
        self.anomaly_analyse.sorted_anomalies = self.anomalies
        with patch("msprobe.pytorch.monitor.anomaly_analyse.check_file_or_directory_path", return_value=None):
            self.anomaly_analyse.rewrite_sorted_anomalies(output_path)

        # 验证调用
        mock_get_anomaly_dict.assert_called_once_with(self.anomaly_analyse.sorted_anomalies)
        mock_save_json.assert_called_once_with(
            os.path.join(output_path, 'anomaly_analyse.json'), 
            {'anomalies': 'data'}, 
            indent=1
        )
        mock_logger.info.assert_called_once_with("anomaly_analyse.json is at output_path.")

    @patch('msprobe.pytorch.monitor.anomaly_analyse.os.path.exists')
    @patch('msprobe.pytorch.monitor.anomaly_analyse.logger')
    def test_rewrite_sorted_anomalies_file_exists(self, mock_logger, mock_exists):
        # 模拟文件已经存在的情况
        mock_exists.return_value = True
        output_path = 'output_path'

        # 调用方法
        with patch("msprobe.pytorch.monitor.anomaly_analyse.check_file_or_directory_path", return_value=None), \
            patch("msprobe.pytorch.monitor.anomaly_analyse.remove_path", return_value=None), \
            patch("msprobe.pytorch.monitor.anomaly_analyse.save_json", return_value=None):
            self.anomaly_analyse.rewrite_sorted_anomalies(output_path)

        # 验证日志警告
        mock_logger.warning.assert_called_once_with(f"The existing file will be deleted: output_path/anomaly_analyse.json.")


class TestParseArgs(unittest.TestCase):

    @patch('msprobe.pytorch.monitor.anomaly_analyse.sys.argv', new=['script_name', '-d', 'path/to/data', '-o', 'path/to/output', '-k', '5', '-s', '[1,2,3]'])
    def test_parse_args_with_all_arguments(self):
        args = _get_parse_args()
        self.assertEqual(args.data_path_dir, 'path/to/data')
        self.assertEqual(args.out_path, 'path/to/output')
        self.assertEqual(args.top_k_number, 5)
        self.assertEqual(args.step_list, '[1,2,3]')

    @patch('msprobe.pytorch.monitor.anomaly_analyse.sys.argv', new=['script_name', '-d', 'path/to/data'])
    def test_parse_args_with_required_argument_only(self):
        args = _get_parse_args()
        self.assertEqual(args.data_path_dir, 'path/to/data')
        self.assertEqual(args.out_path, '')
        self.assertEqual(args.top_k_number, 8)  # 默认值
        self.assertEqual(args.step_list, '[]')  # 默认值

    @patch('msprobe.pytorch.monitor.anomaly_analyse.sys.argv', new=['script_name', '-d', 'path/to/data', '-k', '10'])
    def test_parse_args_with_topk_only(self):
        args = _get_parse_args()
        self.assertEqual(args.data_path_dir, 'path/to/data')
        self.assertEqual(args.out_path, '')
        self.assertEqual(args.top_k_number, 10)  # 提供的值
        self.assertEqual(args.step_list, '[]')  # 默认值


class TestGetStepAndStop(unittest.TestCase):

    def test_valid_step_list_and_top_k(self):
        # 构造有效的 args 对象
        args = MagicMock()
        args.step_list = '[1, 2, 3]'
        args.top_k_number = 5

        step_list, top_k = _get_step_and_stop(args)

        self.assertEqual(step_list, [1, 2, 3])
        self.assertEqual(top_k, 5)

    def test_invalid_step_list(self):
        # 构造无效的 args 对象
        args = MagicMock()
        args.step_list = '[1, 2, 3'  # 不完整的列表
        args.top_k_number = 5

        with self.assertRaises(Exception) as context:
            _get_step_and_stop(args)

        self.assertEqual(str(context.exception), "The step list must be a resolvable list type.")

    def test_non_list_step_list(self):
        # 构造无效的 args 对象
        args = MagicMock()
        args.step_list = 'not_a_list'  # 非列表
        args.top_k_number = 5

        with self.assertRaises(Exception) as context:
            _get_step_and_stop(args)

        self.assertEqual(str(context.exception), "The step list must be a resolvable list type.")

    def test_top_k_number_zero(self):
        # 构造无效的 args 对象
        args = MagicMock()
        args.step_list = '[1, 2, 3]'
        args.top_k_number = 0  # 非法值

        with self.assertRaises(Exception) as context:
            _get_step_and_stop(args)

        self.assertEqual(str(context.exception), "The top k number must be greater than 0.")

    def test_top_k_number_negative(self):
        # 构造无效的 args 对象
        args = MagicMock()
        args.step_list = '[1, 2, 3]'
        args.top_k_number = -1  # 非法值

        with self.assertRaises(Exception) as context:
            _get_step_and_stop(args)

        self.assertEqual(str(context.exception), "The top k number must be greater than 0.")


class TestAnomalyAnalyseFunction(unittest.TestCase):

    @patch('msprobe.pytorch.monitor.anomaly_analyse._get_parse_args')  # 模拟命令行参数解析
    @patch('msprobe.pytorch.monitor.anomaly_analyse._get_step_and_stop')  # 模拟步骤和顶级数字解析
    @patch('msprobe.pytorch.monitor.anomaly_analyse.AnomalyDataLoader')  # 模拟数据加载器
    @patch('msprobe.pytorch.monitor.anomaly_analyse.AnomalyAnalyse')  # 模拟异常分析器
    @patch('msprobe.pytorch.monitor.anomaly_analyse.logger')  # 模拟日志记录
    def test_anomaly_analyse(self, mock_logger, mock_anomaly_analyse, mock_anomaly_data_loader, mock_get_step_and_stop, mock_get_parse_args):
        # 模拟命令行参数
        mock_args = MagicMock()
        mock_args.data_path_dir = 'path/to/data'
        mock_args.out_path = 'path/to/output'
        mock_args.step_list = '[1, 2, 3]'
        mock_args.top_k_number = 5
        mock_get_parse_args.return_value = mock_args

        # 模拟步骤和顶级数字
        mock_step_list = [1, 2, 3]
        mock_top_k_number = 5
        mock_get_step_and_stop.return_value = (mock_step_list, mock_top_k_number)

        # 模拟数据加载
        mock_loader_instance = MagicMock()
        mock_loader_instance.get_anomalies_from_jsons.return_value = [
            MagicMock(message='Anomaly 1'),
            MagicMock(message='Anomaly 2'),
            MagicMock(message='Anomaly 3')
        ]
        mock_anomaly_data_loader.return_value = mock_loader_instance

        # 模拟异常分析
        mock_analyser_instance = MagicMock()
        mock_analyser_instance.get_range_top_k.return_value = [
            MagicMock(message='Top Anomaly 1'),
            MagicMock(message='Top Anomaly 2')
        ]
        mock_anomaly_analyse.return_value = mock_analyser_instance

        # 调用被测试的函数
        _anomaly_analyse()

        # 验证调用
        mock_get_parse_args.assert_called_once()
        mock_get_step_and_stop.assert_called_once_with(mock_args)
        mock_anomaly_data_loader.assert_called_once_with(mock_args.data_path_dir)
        mock_loader_instance.get_anomalies_from_jsons.assert_called_once()
        mock_analyser_instance.get_range_top_k.assert_called_once_with(
            mock_top_k_number, mock_step_list, mock_loader_instance.get_anomalies_from_jsons.return_value
        )
        mock_analyser_instance.rewrite_sorted_anomalies.assert_called_once_with(mock_args.out_path)

        # 验证日志记录
        mock_logger.info.assert_any_call(f"Top {mock_top_k_number} anomalies are listed as follows:")
        mock_logger.info.assert_any_call("0: Top Anomaly 1")
        mock_logger.info.assert_any_call("1: Top Anomaly 2")
