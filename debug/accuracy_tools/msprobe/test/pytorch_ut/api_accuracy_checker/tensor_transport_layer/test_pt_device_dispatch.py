# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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

import unittest
from unittest.mock import MagicMock, patch

from msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.device_dispatch import run_ut_process, \
    online_precision_compare, online_compare, ConsumerDispatcher
from msprobe.pytorch.common.log import logger


class TestDeviceDispatchFunc(unittest.TestCase):
    @patch("msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.device_dispatch.online_compare")
    @patch("msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.device_dispatch.torch")
    def test_run_ut_process(self, mock_torch, mock_online_compare):
        xpu_id = 1
        mock_consumer_queue = MagicMock()
        mock_consumer_queue.empty.side_effect = [True, False, False]
        mock_api_data = MagicMock()
        mock_api_data.name.split.return_value = ("test", "conv2d", 1)
        mock_consumer_queue.get.side_effect = [mock_api_data, "KILL_"]

        run_ut_process(xpu_id, mock_consumer_queue, None, None)
        mock_torch.device.assert_called_once_with('cuda:1')
        mock_online_compare.assert_called_with(mock_api_data, mock_torch.device(), None)

    @patch("msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.device_dispatch.UtDataInfo")
    @patch("msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.device_dispatch.exec_api")
    @patch("msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.device_dispatch.generate_cpu_params")
    def test_online_precision_compare(self, mock_gen_cpu_params, mock_exec_api, mock_ut_data_info):
        with patch("msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.device_dispatch.move2target_device"), \
                patch("msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.device_dispatch.pd"), \
                patch(
                    "msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.device_dispatch.online_api_precision_compare"):
            mock_gen_cpu_params.return_value = (MagicMock(), MagicMock())
            mock_api_data = MagicMock()
            mock_api_data.name.split.return_value = ("tensor", "conv2d", 1)
            mock_com_config = MagicMock()
            mock_api_precision_csv_file = [MagicMock(), MagicMock()]
            online_precision_compare(mock_api_data, None, mock_com_config, mock_api_precision_csv_file)
            mock_gen_cpu_params.assert_called()
            mock_exec_api.assert_called()
            mock_ut_data_info.assert_called()

    @patch.object(logger, "info")
    @patch("msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.device_dispatch.move2target_device")
    def test_online_compare_success(self, mock_move2target_device, mock_logger_info):
        api_data = MagicMock()
        api_data.name = "test_api_name"
        common_config = MagicMock()
        common_config.compare.compare_output.return_value = ("test_fwd_success", "test_bwd_success")
        online_compare(api_data, None, common_config)
        mock_move2target_device.assert_called()
        mock_logger_info.assert_called_once_with("running api_full_name test_api_name ut, "
                                                 "is_fwd_success: test_fwd_success, "
                                                 "is_bwd_success: test_bwd_success")

    @patch.object(logger, "error")
    @patch("msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.device_dispatch.move2target_device")
    def test_online_compare_failed(self, mock_move2target_device, mock_logger_error):
        api_data = MagicMock()
        api_data.name.split.return_value = ["tensor", "conv2d", 1]
        common_config = MagicMock()
        online_compare(api_data, None, common_config)
        mock_move2target_device.assert_called()
        mock_logger_error.assert_called()


class TestConsumerDispatcher(unittest.TestCase):
    @patch("msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.device_dispatch.mp")
    def setUp(self, mock_mq):
        self.mock_mq = mock_mq
        self.consumer_dispatcher = ConsumerDispatcher(None)

    @patch.object(logger, "info")
    @patch("msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.device_dispatch.mp")
    @patch("msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.device_dispatch.CommonCompareConfig")
    def test_start(self, mock_com_compare_config, mock_mq, mock_log_info):
        self.consumer_dispatcher.start(None, None)
        mock_com_compare_config.assert_called_once_with(None, None, None)
        mock_mq.Process.assert_called()
        mock_log_info.assert_any_call("Successfully start unittest process.")

    @patch.object(logger, "info")
    def test_stop(self, mock_log_info):
        mock_queue = MagicMock()
        mock_queue.full.side_effect = [True, False]
        self.consumer_dispatcher.queues = [mock_queue]

        mock_process = MagicMock()
        self.consumer_dispatcher.processes = [mock_process]
        self.consumer_dispatcher.stop()
        mock_log_info.assert_any_call("Successfully stop unittest process.")
        mock_process.join.assert_called()

    def test_update_consume_queue(self):
        self.consumer_dispatcher._choose_max_empty_site_strategy = MagicMock()
        self.consumer_dispatcher._choose_max_empty_site_strategy.return_value = 0
        mock_queue = MagicMock()
        self.consumer_dispatcher.queues = [mock_queue]
        self.consumer_dispatcher.update_consume_queue("test_data")
        mock_queue.put.assert_called_once_with("test_data")

    def test_choose_max_empty_site_strategy(self):
        mock_queue = MagicMock()
        mock_queue.qsize.return_value = 1
        self.consumer_dispatcher.queues = [mock_queue]
        self.consumer_dispatcher.capacity = 5
        self.consumer_dispatcher.reverse_sort = False
        result = self.consumer_dispatcher._choose_max_empty_site_strategy()
        self.assertEqual(result, 0)
