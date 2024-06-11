import time

import torch
import torch.multiprocessing as mp

from api_accuracy_checker.tensor_transport_layer.attl import move2target_device
from api_accuracy_checker.common.utils import print_error_log, print_warn_log, \
    print_info_log, logger, Const


def run_ut_process(xpu_id, compare, consumer_queue, func, config):
    device = torch.device(f'cuda:{xpu_id}')

    while True:
        if consumer_queue.empty():
            time.sleep(0.1)
            continue

        api_data = consumer_queue.get()
        if api_data == "KILL_":
            return

        api_full_name = api_data.name
        api_data = move2target_device(api_data, device)
        try:
            data_info = func(api_full_name, api_data, config.backward_content)
            logger.debug(f"success exec in device {api_full_name}")
            is_fwd_success, is_bwd_success = compare.compare_output(api_full_name, data_info)
            print_info_log(f"running api_full_name {api_full_name} ut, "
                           f"is_fwd_success: {is_fwd_success}, "
                           f"is_bwd_success: {is_bwd_success}")
        except Exception as err:
            [_, api_name, _] = api_full_name.split(Const.DELIMITER)
            if "expected scalar type Long" in str(err):
                print_warn_log(f"API {api_name} not support int32 tensor in CPU, please add {api_name} to CONVERT_API "
                               f"'int32_to_int64' list in accuracy_tools/api_accuracy_check/common/utils.py file.")
            else:
                print_error_log(f"Run {api_full_name} UT Error: {str(err)}")

            compare.write_summary_csv((api_full_name, "SKIP", "SKIP", str(err), api_data.rank))

        finally:
            torch.cuda.empty_cache()


class ConsumerDispatcher:
    def __init__(self, compare, capacity=10, num_workers=8, device: str = "gpu") -> None:
        self.num_workers = num_workers
        self.capacity = capacity
        self.compare = compare
        self.queues = []
        self.reverse_sort = False
        self.pool = None
        self.device = device
        self.data_id = 0
        self.lock = mp.Lock()
        self.result_queue = mp.Queue()
        mp.set_start_method("spawn", force=True)

    def start(self, handle_func, config):
        self.processes = []
        self.queues = [mp.Queue(maxsize=self.capacity) for _ in range(self.num_workers)]
        for xpu_id, q in enumerate(self.queues):
            p = mp.Process(name="run_ut_process", target=run_ut_process,
                           args=(xpu_id, self.compare, q, handle_func, config))

            p.start()
            self.processes.append(p)
        print_info_log("Successfully start unittest process.")

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

    def stop(self):
        for q in self.queues:
            while q.full():
                time.sleep(0.1)
            q.put("KILL_")

        for p in self.processes:
            p.join()
        print_info_log("Successfully stop unittest process.")
