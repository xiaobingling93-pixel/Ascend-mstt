#! /bin/bash

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


# Local envs
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0

# Also indicates NPU per node
NGPUS_PER_NODE=8

export HCCL_WHITELIST_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPU_ASD_ENABLE=0
export ASCEND_LAUNCH_BLOCKING=1

# set hccl timeout time in seconds 不能小于120
export HCCL_CONNECT_TIMEOUT=128

DATE=$(date '+%m%d%H%M%S')
PROFILING_PATH=$1
SCRIPT_PATH=$(realpath "$0")
PARENT_PATH=$(dirname "$SCRIPT_PATH")
export PYTHONPATH=$(dirname $(dirname "$PARENT_PATH")):$PYTHONPATH

echo "--------------------current PROFILING_PATH=$PROFILING_PATH"

mkdir -p ${PROFILING_PATH}
FILE_NAME=${PROFILING_PATH}/p2p_intra_node.csv
LOG_PATH="./logs"
mkdir $LOG_PATH

MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
NNODES=1 \
GPUS_PER_NODE=2 \
NODE_RANK=$NODE_RANK \
FILE_NAME=$FILE_NAME \
python3 ${PARENT_PATH}/p2p_band_profiler.py 2>&1 | tee $LOG_PATH/profiler_$DATE.log