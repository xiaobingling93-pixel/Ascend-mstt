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

if [[ $(pwd) == *"ma-user"* ]]; then
    source /home/ma-user/Ascend/ascend-toolkit/set_env.sh
fi

# 1. 命令行入参校验 若异常则提示检查`profile_space.py`
if [ "$#" -lt 4 ]; then
    echo "Error: Script profile.sh requires at least 4 arguments, but get $# arguments"
    echo "       Supposed arguments: model_name model_size TP SP EP=0 mbs_limit=65536 save_path=./profiled_data suffix=DateTimeStamp version=1.1"
    echo "       Please check TinkerScripter.run_config() in profile_space.py"
    exit 1
fi
SUFFIX=-${8:-"$(date '+%y%m%d-%H%M%S')"}
if [ $SUFFIX == "-" ]; then
    SUFFIX=""
fi
RUNTIME_PATH=${7:-"$(pwd)/profiled_data"}
mbs_limit=${6:-65536}
ep=${5:-0}

export ML_VERSION=${9:-1.1}
export IS_TUNE=${10:-0}

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_PATH=$(dirname $(dirname "$SCRIPT_DIR"))
export PYTHONPATH=$PROJECT_PATH:$PYTHONPATH

# 2. 变量初始化，其中系统变量的export相关逻辑在各model.sh中完成
find_free_port() {
  local port=7000
  while netstat -an | grep -q "$port"; do
    port=$((port + 1))
  done
  echo $port
}

MASTER_ADDR=localhost
MASTER_PORT=$(find_free_port)
echo 使用端口 $MASTER_PORT
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
"

# 可调节的profiler超参，目前看取3-10和10-40无影响
WARMUP_TIMES=3
REPEAT_TIMES=10

MAX_NUM_GPUS=8

# 3. 模型结构参数脚本，读取模型结构命令行参数
ascend_model_script="$SCRIPT_DIR/ascendcloud_model_${1}_${2}.sh"

model_script="$SCRIPT_DIR/model_${1}_${2}.sh"

# 检查脚本文件是否存在
if [ -f "$model_script" ]; then
    effect_script=$model_script
elif [ -f "$ascend_model_script" ]; then
    effect_script=$ascend_model_script
else
    echo "Error: Script '$model_script' or '$ascend_model_script' not found."
    exit 1
fi

# 为不同环境生成存放词表和数据文件的路径
if [ -z "$ML_MODEL_PATH" ]; then
    CURRENT_PATH=$(pwd)
    if [[ $CURRENT_PATH == *"ma-user"* ]]; then
        export ML_MODEL_PATH="/home/ma-user/work/modellink-resources"
    else
        export ML_MODEL_PATH="."
    fi
fi

# 待覆盖变量
GPT_ARGS=""
MOE_ARGS=""

# 此脚本应使能 WITHOUT_JIT_COMPILE TOKENIZER_PATH GPT_ARGS MOE_ARGS 的刷新
echo "source $effect_script"
source $effect_script

# 此处代码和effect_script有顺序关系
MODEL_NAME=$1
MODEL_SIZE=$2

# 4. 数据落盘地址
PROFILING_PATH="${RUNTIME_PATH}/profiled-data-${MODEL_NAME}-${MODEL_SIZE}${SUFFIX}"  # 若目录不存在，则会自动创建
if [ "$#" -lt 8 ]; then
    rm -rf $PROFILING_PATH  # 未指定时，才删除重复拉起的目录，但目前没用
fi
mkdir -p ${PROFILING_PATH}

PROF_ARGS="
    --prof-path ${PROFILING_PATH} \
    --prof-model-name ${MODEL_NAME} \
    --prof-model-size ${MODEL_SIZE} \
    --prof-warmup-times ${WARMUP_TIMES} \
    --prof-repeat-times ${REPEAT_TIMES} \
"

torch_run() {
    local tp=$1
    local sp=$2
    local ep=$3
    local mbs_limit=$4
    local dp=1
    if ((ep >= 1)); then
        let dp=ep
    fi
    if ((tp * dp > MAX_NUM_GPUS || tp == 1 && sp == 1)); then
        return 1
    fi
    EXP_ID="tp${tp}_sp${sp}_ep${ep}"
    echo "================================ working on ${EXP_ID} ================================"
    let gpu=tp*dp
    SUMMARIZE_ARGS="
        ${PROF_ARGS}
        ${GPT_ARGS}
        --tensor-model-parallel-size ${tp}
        --pipeline-model-parallel-size 1
        --distributed-timeout-minutes 5
    "
    if [ "${ep}" -ge 1 ]; then
        SUMMARIZE_ARGS="${SUMMARIZE_ARGS} ${MOE_ARGS} --expert-model-parallel-size ${ep}"
    fi
    if [ "${sp}" -eq 1 ]; then
        SUMMARIZE_ARGS="${SUMMARIZE_ARGS} --sequence-parallel"
    fi
    # 可规避一部分mbs oom情况
    SUMMARIZE_ARGS="${SUMMARIZE_ARGS} --prof-mbs-limit ${mbs_limit}"
    echo [TIME] before profiling ${EXP_ID} : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}/profiling_${MODEL_NAME}.log

    torchrun ${DISTRIBUTED_ARGS} --nproc_per_node ${gpu} $SCRIPT_DIR/block_profiler.py \
        ${SUMMARIZE_ARGS} \
        2>&1 | tee ${PROFILING_PATH}/profiling_${MODEL_NAME}_${EXP_ID}.log

    echo [TIME] after profiling ${EXP_ID} : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}/profiling_${MODEL_NAME}.log
}

# 6. 拉起该次profiler任务: tp sp ep mbs_limit
torch_run $3 $4 $ep $mbs_limit