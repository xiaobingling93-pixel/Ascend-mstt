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

import os
import json
import warnings
from typing import Optional


def get_pytorch_rank_id() -> Optional[int]:
    """Get pytorch rank id."""
    try:
        import torch
        rank_id = os.environ.get("RANK")
        if rank_id is None and torch.distributed.is_available() and torch.distributed.is_initialized():
            rank_id = torch.distributed.get_rank()
        if rank_id is not None and not isinstance(rank_id, int):
            rank_id = int(rank_id)
    except Exception as ex:
        raise RuntimeError(f"Get rank id failed in pytorch: {str(ex)}") from ex
    return rank_id


def get_pytorch_parallel_group_info() -> str:
    """Get pytorch parallel group info."""
    try:
        import torch
        from torch.distributed.distributed_c10d import _world as distributed_world
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            group_info = {}
            global_rank = torch.distributed.get_rank()
            for group in distributed_world.pg_map.keys():
                if torch.distributed.get_backend(group) != "hccl":
                    continue
                hccl_group = group._get_backend(torch.device("npu"))
                comm_name = hccl_group.get_hccl_comm_name(global_rank, init_comm=False)
                if comm_name:
                    group_info[comm_name] = {
                        "group_name": hccl_group.options.hccl_config.get("group_name", ""),
                        "group_rank": torch.distributed.get_group_rank(group, global_rank),
                        "global_ranks": torch.distributed.get_process_group_ranks(group)
                    }
            default_group = torch.distributed.distributed_c10d._get_default_group()
            comm_name = default_group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank, init_comm=False)
            if comm_name:
                group_info[comm_name] = {
                    "group_name": "default_group",
                    "group_rank": torch.distributed.get_group_rank(default_group, global_rank),
                    "global_ranks": torch.distributed.get_process_group_ranks(default_group)
                }
            if group_info:
                return json.dumps(group_info)
    except Exception as ex:
        raise RuntimeError(f"Get parallel group info in pytorch failed: {str(ex)}.") from ex
    return ""


def get_mindspore_rank_id() -> Optional[int]:
    """Get mindspore rank id."""
    try:
        import mindspore.communication as comm
        rank_id = os.environ.get("RANK_ID")
        if rank_id is None and comm.GlobalComm.INITED:
            rank_id = comm.get_rank()
        if rank_id is not None and not isinstance(rank_id, int):
            rank_id = int(rank_id)
    except Exception as ex:
        raise RuntimeError(f"Get rank id failed in mindspore: {str(ex)}") from ex
    return rank_id


def get_mindspore_parallel_group_info() -> str:
    """Get mindspore parallel group info."""
    try:
        import mindspore.communication as comm
        import mindspore.communication._comm_helper as comm_helper
        if comm.GlobalComm.INITED and comm.GlobalComm.BACKEND == comm_helper.Backend.HCCL:
            group_info = {}
            for group_name in comm_helper._get_group_map().keys():
                comm_name = comm.get_comm_name(group_name)
                if not comm_name:
                    continue
                group_info[comm_name] = {
                    "group_name": group_name,
                    "group_rank": comm.get_local_rank(group_name),
                    "global_ranks": comm.get_process_group_ranks(group_name)
                }
            if group_info:
                return json.dumps(group_info)
    except Exception as ex:
        raise RuntimeError(f"Get parallel group info in mindspore failed: {str(ex)}.") from ex
    return ""


def get_rank_id() -> int:
    """Get rank id."""
    rank_id = None
    try:
        rank_id = get_pytorch_rank_id()
    except Exception as ex:
        warnings.warn(f"{str(ex)}")

    if rank_id is None:
        try:
            rank_id = get_mindspore_rank_id()
        except Exception as ex:
            warnings.warn(f"{str(ex)}")

    if rank_id is None:
        warnings.warn("Failed to get rank id from pytorch and mindspore, set rank id to -1.")
        rank_id = -1

    return rank_id


def get_parallel_group_info() -> str:
    """Get parallel group info."""
    parallel_group_info = ""
    try:
        parallel_group_info = get_pytorch_parallel_group_info()
    except Exception as ex:
        warnings.warn(f"{str(ex)}")

    if not parallel_group_info:
        try:
            parallel_group_info = get_mindspore_parallel_group_info()
        except Exception as ex:
            warnings.warn(f"{str(ex)}")

    return parallel_group_info
