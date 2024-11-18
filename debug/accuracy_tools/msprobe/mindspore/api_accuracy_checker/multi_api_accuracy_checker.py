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

# 标准库导入
import multiprocessing
from multiprocessing import Manager
import os
import signal
import sys
import time

# 第三方库导入
from mindspore import context
import numpy as np
from tqdm import tqdm

# 本地应用/库特定导入
from msprobe.core.common.const import Const, CompareConst, MsCompareConst
from msprobe.mindspore.api_accuracy_checker.api_accuracy_checker import ApiAccuracyChecker, BasicInfoAndStatus
from msprobe.mindspore.api_accuracy_checker.multi_data_manager import MultiDataManager
from msprobe.mindspore.common.log import logger


def handle_child_signal(signum, frame):
    """处理子进程的退出信号"""
    logger.info(f"Child process received signal {signum} with PID {os.getpid()}")


def handle_main_signal(signum, frame):
    """处理主进程的退出信号"""
    logger.info(f"Main process received signal {signum} with PID {os.getpid()}")


class MultiApiAccuracyChecker(ApiAccuracyChecker):
    def __init__(self, args):
        # 可以添加 MultiApiAccuracyChecker 特有的属性或方法
        self.api_infos = dict()

        # 使用 Manager 创建共享变量，确保进程间的同步
        self.manager = Manager()
        self.is_first_write = self.manager.Value('b', True)  # 创建共享变量

        # 初始化 DataManager 时传入共享的 is_first_write
        self.multi_data_manager = MultiDataManager(args.out_path, args.result_csv_path, self.is_first_write)

        self.args = args  # 将 args 保存为类的属性


    def process_on_device(self, device_id, api_infos, progress_queue):
        # 设置 MindSpore context 的 device_id
        context.set_context(device_id=device_id)

        # 遍历当前进程分配的任务
        for idx, (api_name_str, api_info) in enumerate(api_infos):
            logger.debug(f"Processing API: {api_name_str}, Device: {device_id}")

            if not self.multi_data_manager.is_unique_api(api_name_str):
                logger.debug(f"API {api_name_str} is not unique, skipping.")
                progress_queue.put(1)
                continue

            # 处理前向
            forward_output_list = self.process_forward(api_name_str, api_info)
            if forward_output_list is not None:
                self.multi_data_manager.record(forward_output_list)

            # 处理反向
            backward_output_list = self.process_backward(api_name_str, api_info)
            if backward_output_list is not None:
                self.multi_data_manager.record(backward_output_list)

            # 保存结果
            self.multi_data_manager.save_results(api_name_str)
            progress_queue.put(1)  # 更新进度

    def run_and_compare(self):
        # 获取要使用的设备ID列表
        device_ids = self.args.device_id

        # 按设备数划分要处理的 API 项
        partitioned_api_infos = list(self.api_infos.items())

        # 在主进程中进行交叉任务切分（基于取模的方式）
        partitioned_api_infos_split = [[] for _ in range(len(device_ids))]
        for idx, api_info in enumerate(partitioned_api_infos):
            device_index = idx % len(device_ids)  # 使用取模方法分配任务
            partitioned_api_infos_split[device_index].append(api_info)

        # 创建一个共享进度队列
        progress_queue = multiprocessing.Queue()

        # 进度条
        total_tasks = len(partitioned_api_infos)  # 计算总任务数
        with tqdm(total=total_tasks, desc="Total Progress", ncols=100) as pbar:
            # 创建多进程
            processes = []
            for index, device_id in enumerate(device_ids):
                process = multiprocessing.Process(target=self.process_on_device,
                                                args=(device_id, partitioned_api_infos_split[index], progress_queue))
                processes.append(process)
                process.start()

            # 主进程更新进度条
            completed_tasks = 0
            while completed_tasks < total_tasks:
                try:
                    completed_tasks += progress_queue.get(timeout=60)  # 设置超时时间（秒）
                    pbar.update(1)
                except multiprocessing.queues.Empty:
                    logger.error("Timeout while waiting for progress updates. Skipping remaining tasks.")
                    break

                # 检查子进程状态
                for process in processes:
                    if not process.is_alive():
                        if process.exitcode != 0:
                            logger.error(f"Process {process.pid} exited with code {process.exitcode}.")
                            total_tasks -= len(partitioned_api_infos_split[processes.index(process)])
                        processes.remove(process)


            # 确保所有子进程完成或终止
            for process in processes:
                process.join(timeout=60)
                if process.is_alive():
                    logger.error(f"Process {process.pid} did not terminate. Forcing termination.")
                    process.terminate()
