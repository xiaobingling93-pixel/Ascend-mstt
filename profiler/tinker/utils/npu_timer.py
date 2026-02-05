# Copyright (c) 2025, Huawei Technologies Co., Ltd.
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

import torch


class NPUTimer:
    def __init__(self):
        # 在初始化时创建开始和结束事件
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.elapsed_times = []

    def start(self):
        """开始计时。"""
        # 记录开始事件
        self.start_event.record()

    def stop(self):
        """停止计时，并记录耗时。"""
        # 记录结束事件
        self.end_event.record()
        # 同步，确保事件已记录
        torch.cuda.synchronize()
        # 计算耗时并保存
        elapsed_time = self.start_event.elapsed_time(self.end_event) * 1000
        self.elapsed_times.append(elapsed_time)

    def reset(self):
        """重置计时器，清空所有记录的时间。"""
        self.elapsed_times = []

    def get_times(self):
        """获取所有记录的耗时。"""
        return self.elapsed_times

    def get_total_time(self):
        """获取总耗时。"""
        return sum(self.elapsed_times)

    def get_average_time(self):
        """获取平均耗时。"""
        if self.elapsed_times:
            return sum(self.elapsed_times) / len(self.elapsed_times)
        else:
            return 0.0
