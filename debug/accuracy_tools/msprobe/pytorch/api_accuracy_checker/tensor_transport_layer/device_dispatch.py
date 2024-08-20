import time

import pandas as pd
import torch
import torch.multiprocessing as mp

from msprobe.core.common.const import Const
from msprobe.pytorch.api_accuracy_checker.compare.api_precision_compare import online_api_precision_compare
from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import DETAIL_TEST_ROWS, thousandth_standard_api, \
    binary_standard_api, absolute_standard_api
from msprobe.pytorch.api_accuracy_checker.run_ut.run_ut_utils import UtDataInfo, exec_api
from msprobe.pytorch.common.utils import logger
from msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.attl import move2target_device

# NPU vs GPU api list
CompareApi = set(absolute_standard_api) | set(binary_standard_api) | set(thousandth_standard_api)


def run_ut_process(xpu_id, compare, consumer_queue, func, config):
    """ When consumer_queue(shared with ConsumerDispatcher) is not empty, consume api data from consumer_queue.
    :param xpu_id: int
    :param compare: instance of Comparator
    :param consumer_queue: shared queues of ConsumerDispatcher
    :param func: run_touch_api_online
    :param config: run_ut_config
    :return:
    """
    gpu_device = torch.device(f'cuda:{xpu_id}')

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
            online_compare(api_data, gpu_device, compare, func, config)
        else:
            # NPUvsCPU vs GPUvsCPU
            online_precision_compare(api_data, gpu_device, compare, func, config)


def online_precision_compare(api_data, device, compare, func, config):
    """online run_ut for precision_compare: NPUvsCPU vs GPUvsCPU
    1. get NPUvsCPU compare result
    2. get GPUvsCPU compare result
    3. call online_api_precision_compare
    """

    api_full_name = api_data.name
    [api_type, api_name, _] = api_full_name.split(Const.SEP)
    npu_args, npu_kwargs, npu_out = api_data.args, api_data.kwargs, api_data.result

    if npu_kwargs.get("device"):
        del npu_kwargs["device"]

    try:
        # NPU vs CPU
        cpu_out = exec_api(api_type, api_name, npu_args, npu_kwargs)
        npu_data_info = UtDataInfo(None, None, npu_out, cpu_out, None, [], None, rank=api_data.rank)
        logger.debug(f"success exec run_ut in cpu device {api_full_name}")
        npu_detail = compare.compare_output(api_full_name, npu_data_info, True)
        npu_data = pd.DataFrame(npu_detail, columns=DETAIL_TEST_ROWS[-1])

        # GPU vs CPU
        api_data_gpu = move2target_device(api_data, device)  # args, kwargs -> gpu, result -> npu
        data_info = func(api_full_name, api_data_gpu, config.backward_content)
        gpu_out = data_info.bench_output
        gpu_data_info = UtDataInfo(None, None, gpu_out, cpu_out, None, [], None, rank=api_data.rank)
        logger.debug(f"success exec run_ut in gpu device {api_full_name}")
        gpu_detail = compare.compare_output(api_full_name, gpu_data_info, True)
        gpu_data = pd.DataFrame(gpu_detail, columns=DETAIL_TEST_ROWS[-1])

        # NPUvsCPU vs GPUvsCPU
        online_api_precision_compare(npu_data, gpu_data, api_data.rank)

    except Exception as err:
        if "expected scalar type Long" in str(err):
            logger.warning(
                f"API {api_name} not support int32 tensor in CPU, please add {api_name} to CONVERT_API "
                f"'int32_to_int64' list in accuracy_tools/msprobe/core/common/const.py file.")
        elif api_type in [Const.DISTRIBUTED]:
            logger.info(f"{api_full_name} is not supported for run ut. SKIP.")
        else:
            logger.error(f"Run {api_full_name} UT Error: {str(err)}")

        compare.write_summary_csv((api_full_name, "SKIP", "SKIP", [[str(err)]], api_data.rank))

    finally:
        torch.cuda.empty_cache()


def online_compare(api_data, device, compare, func, config):
    """online run_ut for compare：NPU vs GPU"""

    api_full_name = api_data.name
    api_data = move2target_device(api_data, device)
    try:
        data_info = func(api_full_name, api_data, config.backward_content)
        logger.debug(f"success exec run_ut in device {api_full_name}")
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

        compare.write_summary_csv((api_full_name, "SKIP", "SKIP", str(err), api_data.rank))

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
        for xpu_id, q in enumerate(self.queues):
            p = mp.Process(name="run_ut_process", target=run_ut_process,
                           args=(xpu_id, self.compare, q, handle_func, config))

            p.start()
            self.processes.append(p)
        logger.info("Successfully start unittest process.")

    def stop(self):
        for q in self.queues:
            while q.full():
                time.sleep(0.1)
            q.put("KILL_")

        for p in self.processes:
            p.join()
        logger.info("Successfully stop unittest process.")

    def update_consume_queue(self, api_data):
        while True:
            index = self._choose_max_empty_site_strategy()
            if index != -1:
                q = self.queues[index]
                q.put(api_data)
                logger.debug(f"将{api_data.name}调度给第{index}个GPU")
                break
            logger.debug("所有的UT队列都已满, 阻塞中")
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
