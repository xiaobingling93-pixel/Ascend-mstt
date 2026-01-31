# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
# `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


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
