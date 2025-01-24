# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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


class FileDesc(object):
    def __init__(self, file_name, dir_path, timestamp=-1):
        self.file_name = file_name
        self.dir_path = dir_path
        self.path = os.path.join(dir_path, file_name)
        self.timestamp = timestamp
        self.idx = 0
        if self.timestamp == -1:
            self.timestamp = os.path.getmtime(self.path)


class NpuDumpFileDesc(FileDesc):
    def __init__(self, file_name, dir_path, timestamp, op_name, op_type, task_id, stream_id=0):
        super(NpuDumpFileDesc, self).__init__(file_name, dir_path, timestamp)
        self.op_name = op_name
        self.op_type = op_type
        self.task_id = task_id
        stream_id = 0 if stream_id is None else int(stream_id)
        self.stream_id = stream_id
        self.idx = dir_path.split(os.sep)[-1]


class DumpDecodeFileDesc(NpuDumpFileDesc):
    def __init__(self, file_name, dir_path, timestamp, op_name, op_type, task_id, anchor_type, anchor_idx):
        super(DumpDecodeFileDesc, self).__init__(file_name, dir_path, timestamp, op_name, op_type, task_id)
        self.type = anchor_type
        self.idx = anchor_idx
