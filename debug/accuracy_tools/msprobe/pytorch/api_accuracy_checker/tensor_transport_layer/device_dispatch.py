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

import time
from collections import namedtuple

import pandas as pd
import torch
import torch.multiprocessing as mp

from msprobe.core.common.const import Const, CompareConst
from msprobe.pytorch.api_accuracy_checker.compare.api_precision_compare import online_api_precision_compare
from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import DETAIL_TEST_ROWS, thousandth_standard_api, \
    binary_standard_api, absolute_standard_api
from msprobe.pytorch.api_accuracy_checker.run_ut.run_ut_utils import UtDataInfo, exec_api, ExecParams
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.attl import move2target_device
from msprobe.pytorch.api_accuracy_checker.run_ut.run_ut_utils import generate_cpu_params

# NPU vs GPU api list
CompareApi = set(absolute_standard_api) | set(binary_standard_api) | set(thousandth_standard_api)

current_time = time.strftime("%Y%m%d%H%M%S")
ONLINE_API_PRECISION_COMPARE_RESULT_FILE_NAME = "api_precision_compare_result_" + current_time + "_rank*.csv"
ONLINE_API_PRECISION_COMPARE_DETAILS_FILE_NAME = "api_precision_compare_details_" + current_time + "_rank*.csv"

OnlineApiPrecisionCompareConfig = namedtuple('OnlineApiPrecisionCompareConfig',
                                             ['npu_data', 'gpu_data', 'rank', 'result_csv_path', 'details_csv_path'])
# namedtuple of [instance of Comparator, func of run_touch_api_online, config of run_ut_config]
CommonCompareConfig = namedtuple('CommonCompareConfig', ['compare', 'handle_func', 'config'])


def get_gpu_device():
    is_gpu = False
    try:
        import torch_npu
    except ImportError:
        is_gpu = True
    return is_gpu


def run_ut_process(xpu_id, consumer_queue, common_config, api_precision_csv_file):
    """ When consumer_queue(shared with ConsumerDispatcher) is not empty, consume api data from consumer_queue.
    :param xpu_id: int
    :param consumer_queue: shared queues of ConsumerDispatcher
    :param common_config: namedtuple of CommonCompareConfig
    :param api_precision_csv_file: list, length is 2, result file name and details file name
    :return:
    """
    device_info = "cuda" if get_gpu_device() else "npu"
    logger.info(f"Start run_ut_process for {device_info} device, rank: {xpu_id}.")
    gpu_device = torch.device(f'{device_info}:{xpu_id}')

    while True:
        if consumer_queue.empty():
            time.sleep(0.1)
            continue

        api_data = consumer_queue.get()
        if api_data == "KILL_":
            # current consumer finish
            return

        _, api_name, _ = api_data.name.split(Const.SEP)
        if api_name in CompareApi:
            # NPU vs GPU
            online_compare(api_data, gpu_device, common_config)
        else:
            # NPUvsCPU vs GPUvsCPU
            online_precision_compare(api_data, gpu_device, common_config, api_precision_csv_file)


def online_precision_compare(api_data, device, common_config, api_precision_csv_file):
    """online run_ut for precision_compare: NPUvsCPU vs GPUvsCPU
    1. get NPUvsCPU compare result
    2. get GPUvsCPU compare result
    3. call online_api_precision_compare
    :param api_data
    :param device
    :param common_config: namedtuple of CommonCompareConfig
    :param api_precision_csv_file: [result_file_name, details_file_name]
    """
    compare, func, config = common_config.compare, common_config.handle_func, common_config.config
    api_full_name = api_data.name
    [api_type, api_name, _] = api_full_name.split(Const.SEP)
    npu_args, npu_kwargs, npu_out = api_data.args, api_data.kwargs, api_data.result

    if npu_kwargs.get("device"):
        del npu_kwargs["device"]

    try:
        # NPU vs CPU
        cpu_params = generate_cpu_params(npu_args, npu_kwargs, False, api_name)
        cpu_args, cpu_kwargs = cpu_params.cpu_args, cpu_params.cpu_kwargs
        cpu_exec_params = ExecParams(api_type, api_name, Const.CPU_LOWERCASE, cpu_args, cpu_kwargs, False, None)
        cpu_out = exec_api(cpu_exec_params)
        npu_data_info = UtDataInfo(None, None, npu_out, cpu_out, None, [], None, rank=api_data.rank)
        npu_detail = compare.compare_output(api_full_name, npu_data_info, True)
        npu_data = pd.DataFrame(npu_detail, columns=DETAIL_TEST_ROWS[-1])

        # GPU vs CPU
        api_data_gpu = move2target_device(api_data, device)  # args, kwargs -> gpu, result -> npu
        data_info = func(api_full_name, api_data_gpu, config.backward_content)
        gpu_out = data_info.bench_output
        gpu_data_info = UtDataInfo(None, None, gpu_out, cpu_out, None, [], None, rank=api_data.rank)
        gpu_detail = compare.compare_output(api_full_name, gpu_data_info, True)
        gpu_data = pd.DataFrame(gpu_detail, columns=DETAIL_TEST_ROWS[-1])

        # NPUvsCPU vs GPUvsCPU
        result_file_name, details_file_name = api_precision_csv_file
        precision_compare_config = OnlineApiPrecisionCompareConfig(npu_data, gpu_data, api_data.rank,
                                                                   result_file_name, details_file_name)
        online_api_precision_compare(precision_compare_config)

    except Exception as err:
        if "expected scalar type Long" in str(err):
            logger.warning(
                f"API {api_name} not support int32 tensor in CPU, please add {api_name} to CONVERT_API "
                f"'int32_to_int64' list in accuracy_tools/msprobe/core/common/const.py file.")
        elif api_type in [Const.DISTRIBUTED]:
            logger.info(f"{api_full_name} is not supported for run ut. SKIP.")
        else:
            logger.error(f"Run {api_full_name} UT Error: {str(err)}")

        compare.write_summary_csv((api_full_name, CompareConst.SKIP, CompareConst.SKIP, [[str(err)]], api_data.rank))

    finally:
        torch.cuda.empty_cache()


def online_compare(api_data, device, common_config):
    """online run_ut for compare：NPU vs GPU
    """
    compare, func, config = common_config.compare, common_config.handle_func, common_config.config
    api_full_name = api_data.name
    api_data = move2target_device(api_data, device)
    try:
        data_info = func(api_full_name, api_data, config.backward_content)
        is_fwd_success, is_bwd_success = compare.compare_output(api_full_name, data_info)
        logger.info(f"running api_full_name {api_full_name} ut, "
                    f"is_fwd_success: {is_fwd_success}, "
                    f"is_bwd_success: {is_bwd_success}")
    except Exception as err:
        [api_type, api_name, _] = api_full_name.split(Const.SEP)
        if "expected scalar type Long" in str(err):
            logger.warning(
                f"API {api_name} not support int32 tensor in CPU, please add {api_name} to CONVERT_API "
                f"'int32_to_int64' list in accuracy_tools/msprobe/core/common/const.py file.")
        elif api_type in [Const.DISTRIBUTED]:
            logger.info(f"{api_full_name} is not supported for run ut. SKIP.")
        else:
            logger.error(f"Run {api_full_name} UT Error: {str(err)}")

        compare.write_summary_csv((api_full_name, CompareConst.SKIP, CompareConst.SKIP, [[str(err)]], api_data.rank))

    finally:
        torch.cuda.empty_cache()


class ConsumerDispatcher:
    def __init__(self, compare, capacity=10, num_workers=8, device: str = "gpu") -> None:
        self.num_workers = num_workers
        self.capacity = capacity
        self.compare = compare
        self.queues = []
        self.processes = []
        self.reverse_sort = False
        self.pool = None
        self.device = device
        self.data_id = 0
        self.lock = mp.Lock()
        self.result_queue = mp.Queue()
        mp.set_start_method("spawn", force=True)

    def start(self, handle_func, config):
        self.queues = [mp.Queue(maxsize=self.capacity) for _ in range(self.num_workers)]
        api_precision_csv_file = [
            ONLINE_API_PRECISION_COMPARE_RESULT_FILE_NAME,
            ONLINE_API_PRECISION_COMPARE_DETAILS_FILE_NAME
        ]
        common_config = CommonCompareConfig(self.compare, handle_func, config)
        for xpu_id, q in enumerate(self.queues):
            p = mp.Process(name="run_ut_process", target=run_ut_process,
                           args=(xpu_id, q, common_config, api_precision_csv_file))

            p.start()
            self.processes.append(p)
        logger.info(
            f'Api_precision_compare task result will be saved in {ONLINE_API_PRECISION_COMPARE_RESULT_FILE_NAME}')
        logger.info(
            f"Api_precision_compare task details will be saved in {ONLINE_API_PRECISION_COMPARE_DETAILS_FILE_NAME}")
        logger.info("Successfully start unittest process.")

    def stop(self):
        for q in self.queues:
            while q.full():
                time.sleep(0.1)
            q.put("KILL_")

        for p in self.processes:
            p.join()
        logger.info("Successfully stop unittest process.")
        logger.info(f"Api_precision_compare task result is saved in {ONLINE_API_PRECISION_COMPARE_RESULT_FILE_NAME}")
        logger.info(f"Api_precision_compare task details is saved in {ONLINE_API_PRECISION_COMPARE_DETAILS_FILE_NAME}")

    def update_consume_queue(self, api_data):
        while True:
            index = self._choose_max_empty_site_strategy()
            if index != -1:
                q = self.queues[index]
                q.put(api_data)
                break
            time.sleep(0.1)

    def _choose_max_empty_site_strategy(self):
        maximum = 0
        index = -1
        # 充分利用多卡资源，防止任务过多分配给前面的卡
        _reverse = 1 if not self.reverse_sort else -1
        for i, q in enumerate(self.queues[::_reverse]):
            empty_site = self.capacity - q.qsize()
            if empty_site > maximum:
                maximum = empty_site
                index = i
        index = len(self.queues) - index - 1 if index != -1 and self.reverse_sort else index
        self.reverse_sort = not self.reverse_sort
        return index
