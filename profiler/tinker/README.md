# 功能概述

Tinker大模型并行策略自动寻优系统（后称Tinker），根据提供的训练脚本，进行约15分钟的单节点NPU性能测量，推荐高性能并行策略训练脚本。

1. 数据测量：指定待优化的`.sh`训练脚本，`Tinker`通过`profiler`，获取指定模型在当前硬件上，各模型子图在不同并行策略下的开销性能数据；
2. 策略寻优：指定待优化的`.sh`训练脚本以及步骤1测量数据，`Tinker`通过`optimizer`，推荐时间开销最优的并行策略。
3. 获取推荐结果脚本：策略寻优默认将脚本生成在`./results/`下的子文件夹中。
4. 推荐使用一键运行功能，自动执行数据测量&策略寻优。

# 使用前CheckList

1. 配置训练框架相关环境：请确保使用当前环境已安装/配置`CANN套件`、`训练框架依赖库`、`megatron`、`mindspeed加速库`等依赖（Tinker运行需求训练框架能力）；

   | 软件及版本     | 对应分支                           |
   |-----------|--------------------------------|
   | ModelLink 1.0.RC3 (1.2) | git checkout 4ea42a23          |
   | ModelLink 1.0.RC2 (1.1) | git checkout 2b0edd            |
   | ModelLink 1.0.RC1 (1.0) | git checkout 224ae35           |
   | MindSpeed-LLM 1.0.0 | git checkout 969686ff          |
   | MindSpeed-LLM 2.0.0 | git checkout 2.0.0_core_r0.8.0 |
   | MindSpeed-LLM 2.3.0 | git checkout 2.3.0             |

    python, driver, firmware, CANN, torch, torch_npu以及其他依赖安装步骤请参考训练框架（MindSpeed-LLM）readme。  
    备注：ModelLink从1.0.0开始正式更名为MindSpeed-LLM，Tinker支持上述版本的ModelLink/MindSpeed-LLM，建议使用最新版本以获得更好的性能和功能支持。

2. 准备`待优化训练脚本`：配置待优化的`.sh`训练脚本，特别是词表文件（TOKENIZER_PATH或TOKENIZER_MODEL）和数据文件（DATA_PATH）路径配置（Tinker寻优时需要对齐用户实际训练时的模型（训练配置））；

3. 导入ModelLink（MindSpeed-LLM）框架路径`export ML_PATH=/XX/XX/XX/MindSpeed-LLM`

4. 克隆mstt库`git clone https://gitee.com/ascend/mstt.git`

# 具体使用方式

请通过tinker_auto_parallel入口脚本来使用Tinker的寻优功能，并在工作目录下运行命令和相关参数，建议使用一键运行模式

## 一键运行

【入口】

`python /xxx/tinker/tinker_auto_parallel.py --mode all --config_path XXX.json ...`

【命令示例】请修改入参后运行

```shell
python /xxx/tinker/tinker_auto_parallel.py --mode all --config_path /xxx/tinker/parameter_config.json
python /xxx/tinker/tinker_auto_parallel.py --mode all --pretrain_script_path /xxx/xxx.sh  --output_dir /xxx/results ...
```


- `--mode`：位置参数，推荐使用一键运行(all)功能，此外支持性能测量(profile)，策略寻优(search)，模拟仿真(simulate)三种模式
- 默认全量参数位于 /xxx/tinker/utils/constant.py
- `--config_path`：(可选) 位置参数，用于指定配置文件路径，默认配置文件位于/xxx/tinker/parameter_config.json。使用该参数会覆盖原始路径/xxx/tinker/utils/constant.py。客户可以在config_path里调整参数。
- `--is_full_tune`：布尔值，标记是否为全参微调任务，默认为False，程序会通过脚本路径中关键字`tune` `full`检测，自动赋值，用户无需感知

以下三种模式参数可以在config_path里做修改也可以直接加入命令行

【profile模式参数】
- `--model_name`: 位置参数，用于标识模型名称
- `--model_size`: 位置参数，用于标识模型尺寸
- `--pretrain_script_path`: 指定`待优化脚本`
- `--version`: 指定使用的ModelLink框架版本，可选`1.2` `1.1` `1.0` `1.0.0` `2.0.0`分别对应`1.0.RC3` `1.0.RC2` `1.0.RC1` `1.0.0` `2.0.0`(可选，默认值1.1)
- `--max_npu`: 指定最大npu数(可选，默认值8)

运行后，Tinker会在`./profiled_data`下生成形如`profiled_data_xxx`的测量数据文件夹，内含若干`.csv`文件以及运行日志

【search模式参数】

- `--profiled_data_path`: 指定性能数据所在路径，也可以只传入文件夹名称，Tinker会在`./profiled_data`路径下寻找该文件夹(一键运行模式时不需要该参数)
- `--global_batch_size` `--num_nodes` `--num_npus_per_node`: 分别指定数据批尺寸、节点数量、单节点计算卡数
- `--memory_limit`: 指定内存上限（单位MB，仅算子内存占用，64G内存推荐设定57000，性能摸高时可设定60000；32G内存推荐设定25000，性能摸高时可设定27000）
- `--cpus`: 指定多进程加速计算时使用的进程数(可选，默认值5)
- `--output_dir`: 指定输出结果路径(可选，默认为`./results`)

运行完成后，Tinker会在results文件夹中生成类似`qwen15-7b-gbs32-56000-1nodes8npus-2024-11-18-01-50-38`的文件夹，其中`.log`为运行日志，`config`文件夹存放命令行参数结果。

【simulate模式参数】

- `--profiled_data_path`: 指定性能数据所在路径，也可以只传入文件夹名称，Tinker会在`./profiled_data`路径下寻找该文件
- `--global_batch_size` `--num_nodes` `--num_npus_per_node`: 分别指定数据尺寸、节点数量、单节点计算卡数
- `--simu_tp`: 指定tp值(可选，默认值1)
- `--simu_pp`: 指定pp值(可选，默认值1)
- `--simu_ep`: 指定ep值(可选，默认值1)
- `--simu_sp`: 指定sp值(可选，默认值0)
- `--zero`: 指定ZeRO模式(0关或1开)
- `--micro_batch_size`: 指定micro batch size
- `--num_layer_list`: 模型分层列表，例如4,4,4,4
- `--recompute`: 是否开启重计算(0关或1开)
- `--detail`: 是否展示详细开销信息

【注意】

1. mode、config以外的其它参数均为tinker工具使能的相关运行参数，手动指定后会覆盖parameter_config.json中的默认参数

2. 使能tinker工具时可选择在命令中指定相关参数，也可以选择直接将参数写入parameter_config.json中并指定为config参数对应的配置文件

3. 一键运行模式需要profile模式参数和search模式参数，其余模式只需要对应参数

4. 全参微调脚本使用时，确保脚本名称存在`tune` `full`关键字(如不符合则使能--is_full_tune参数）， 数据集请使用转换后的微调数据集，其余使用流程无变更

【配置文件示例】

```json
{
    "profile": {
      "model_name": null,
      "model_size": null,
      "pretrain_script_path": null,
      "version": "1.1",
      "max_npu": 8
    },
    "search": {
      "profiled_data_path": null,
      "global_batch_size": null,
      "num_nodes": null,
      "num_npus_per_node": 8,
      "cpus": 5,
      "memory_limit": 57000,
      "output_dir": "./results",
      "pretrain_script_path_search": null
    },
    "simulate": {
      "profiled_data_path": null, 
      "num_layers": null,
      "global_batch_size": null,
      "num_nodes": null,
      "num_npus_per_node": 8,
      "simu_tp": 1,
      "simu_pp": 1,
      "simu_ep": 1,
      "simu_sp": 0,
      "zero": 0,
      "micro_batch_size": 1,
      "num_layer_list": null,
      "recompute": 0
    }
  }
```
# FAQ

1. 性能测量中Tinker对训练脚本依赖的说明

   Tinker测量数据时会处理、储存指定的训练脚本并部分运行。请确保正确填写`TOKENIZER_PATH`，`DATA_PATH`（可参考ModelLink ReadMe填写）。Tinker在处理脚本时，会删除torchrun逻辑、并行策略、预训练权重存取相关命令行参数，然后将其他模型、训练相关参数存放于`GPT_ARGS`中。具体剔除的参数包括：
   ```
   --tensor-model-parallel-size
   --pipeline-model-parallel-size
   --sequence-parallel
   --context-parallel-size
   --num-layers-per-virtual-pipeline-stage
   --recompute-xxx
   --use-distributed-optimizer
   --overlap-param-gather
   --num-layer-list
   --save
   --load
   --context-parallel-algo
   --ulysses-degree-in-cp
   --cp-attention-mask-type
   --use-cp-send-recv-overlap
   --kv-head-repeat-before-uly-alltoall
   ```

2. 策略寻优中设定内存上限`-mem`推荐值的解释：当前推荐值为经验值。Tinker仅预测torch算子占用内存开销，因此设定`-mem`时需注意预留CANN、HCCL等组件的内存开销，并避免极端内存使用带来的反复内存搬移带来的性能降低。
3. Tinker在策略寻优完成后可能推荐新的并行策略，此时权重参数可能需要额外转换，请确保脚本中预训练权重参数(ckpt相关参数)匹配新并行策略，且文件路径已正确配置。