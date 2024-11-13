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

import os
import numpy as np
import multiprocessing
from multiprocessing import Manager
from mindspore import context
from tqdm import tqdm

from msprobe.core.common.const import Const, CompareConst, MsCompareConst
from msprobe.mindspore.api_accuracy_checker.api_accuracy_checker import ApiAccuracyChecker, BasicInfoAndStatus
from msprobe.mindspore.api_accuracy_checker.multi_data_manager import MultiDataManager
from msprobe.mindspore.common.log import logger



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

    def process_on_device(self, device_id, partitioned_api_infos, index):
        # 设置 MindSpore context 的 device_id
        context.set_context(device_id=device_id)

        # 使用 numpy.array_split 来均匀分配任务
        partitioned_api_infos_split = np.array_split(partitioned_api_infos, len(self.args.device_id))

        # 获取当前进程要处理的任务
        current_partition = partitioned_api_infos_split[index]

        # 使用 tqdm 进度条，每个进程单独显示
        with tqdm(total=len(current_partition), desc=f"Device {device_id}", position=index) as pbar:
            for idx, (api_name_str, api_info) in enumerate(current_partition):  # 只遍历当前进程分配的任务
                # debug打印每个任务的基本信息
                logger.debug(
                    f"Processing API: {api_name_str}, Index: {idx} (Total tasks: {len(current_partition)}),"
                    f" Device: {device_id}")

                if not self.multi_data_manager.is_unique_api(api_name_str):
                    logger.debug(f"API {api_name_str} is not unique, skipping.")
                    pbar.update(1)
                    continue

                if not api_info.check_forward_info():
                    logger.debug(
                        f"api: {api_name_str} is lack of forward information, skipping forward and backward check.")
                    pbar.update(1)
                    continue

                # 执行前向和后向检查
                try:
                    print(f"Executing forward check for {api_name_str}")
                    forward_inputs_aggregation = self.prepare_api_input_aggregation(api_info, Const.FORWARD)
                    forward_output_list = self.run_and_compare_helper(api_info, api_name_str,
                                                                      forward_inputs_aggregation, Const.FORWARD)
                    self.multi_data_manager.record(forward_output_list)
                except Exception as e:
                    logger.warning(f"Error in forward check for {api_name_str}: {e}")

                if api_info.check_backward_info():
                    try:
                        print(f"Executing backward check for {api_name_str}")
                        backward_inputs_aggregation = self.prepare_api_input_aggregation(api_info, Const.BACKWARD)
                        backward_output_list = self.run_and_compare_helper(api_info, api_name_str,
                                                                           backward_inputs_aggregation, Const.BACKWARD)
                        self.multi_data_manager.record(backward_output_list)
                    except Exception as e:
                        logger.warning(f"Error in backward check for {api_name_str}: {e}")

                # 保存结果
                self.multi_data_manager.save_results(api_name_str)

                # 更新进度条
                pbar.update(1)

    def run_and_compare(self):
        # 获取要使用的设备ID列表
        device_ids = self.args.device_id

        # 按设备数划分要处理的 API 项
        partitioned_api_infos = list(self.api_infos.items())

        # 创建多进程
        processes = []
        for index, device_id in enumerate(device_ids):
            process = multiprocessing.Process(target=self.process_on_device,
                                              args=(device_id, partitioned_api_infos, index))
            processes.append(process)
            process.start()

        # 等待所有进程完成
        for process in processes:
            process.join()


