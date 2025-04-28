# msMonitor: MindStudio一站式在线监控工具

## 安装方式

### 1. clone 代码

```bash
git clone https://gitee.com/ascend/mstt.git
```

### 2. 安装依赖
dynolog的编译依赖，确保安装了以下依赖：
<table>
  <tr>
   <td>Language
   </td>
   <td>Toolchain
   </td>
  </tr>
  <tr>
   <td>C++
   </td>
   <td>gcc 8.5.0+
   </td>
  </tr>
  <tr>
   <td>Rust
   </td>
   <td>Rust >= 1.81
   </td>
  </tr>
</table>

- 安装rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

source $HOME/.cargo/env
```

- 安装ninja

```bash
# debian
sudo apt-get install -y cmake ninja-build

# centos
sudo yum install -y cmake ninja
```

### 3. 编译

- dynolog编译

默认编译生成dyno和dynolog二进制文件, -t参数可以支持将二进制文件打包成deb包或rpm包.

```bash
# 编译dyno和dynolog二进制文件
bash scripts/build.sh

# 编译deb包, 当前支持amd64和aarch64平台, 默认为amd64, 编译aarch64平台需要修改third_party/dynolog/scripts/debian/control文件中的Architecture改为aarch64
bash scripts/build.sh -t deb

# 编译rpm包, 当前只支持amd64平台
bash scripts/build.sh -t rpm
```

- msmonitor_plugin wheel包编译

msmonitor_plugin wheel包提供IPCMonitor，MsptiMonitor等公共能力，使用nputrace和npu-monitor功能前必须安装该wheel包，具体编译安装指导可参考msmonitor\plugin\README.md。

## 使用方式

### Profiler trace dump功能
Profiler trace dump功能基于dynolog开发，实现类似于动态profiling的动态触发Ascend Torch Profiler采集profiling的功能。用户基于dyno CLI命令行可以动态触发指定节点的训练进程trace dump。

- 查看dyno支持的命令和帮助

```bash
dyno --help
```

dyno命令支持的参数选项

| 命令                 | 参数类型   | 说明                           |
|--------------------|--------|------------------------------|
| hostname           | String | 网络中唯一标识一台设备的名称，默认值localhost  |
| port               | i32    | 用于区分同一设备上的不同网络服务或应用程序，默认值1778 |

- 查看nputrace支持的命令和帮助

```bash
dyno nputrace --help
```

- nputrace使用方式

```bash
dyno nputrace [SUBCOMMANDS] --log-file <LOG_FILE>
```

nputrace子命令支持的参数选项

| 子命令 | 参数类型 | 说明 | PyTorch支持 | MindSpore支持 |
|-------|-------|-------|:---------:|:-----------:|
| job-id | u64 | 采集任务的job id，默认值0，dynolog原生参数 |     N     |      N      |
| pids | String | 采集任务的pid列表，多个pid用逗号分隔，默认值0，dynolog原生参数 |     N     |      N      |
| process-limit | u64 | 最大采集进程的数量，默认值3，dynolog原生参数 |     N     |      N      |
| profile-start-time | u64 | 用于同步采集的Unix时间戳，单位毫秒，默认值0，dynolog原生参数 |     N     |      N      |
| duration-ms | u64 | 采集的周期，单位毫秒，默认值500，dynolog原生参数 |     N     |      N      |
| iterations | i64 | 采集总迭代数，默认值-1，dynolog原生参数 |     Y     |      Y      |
| log-file | String | 采集落盘的路径，必选值 |     Y     |      Y      |
| start-step | u64 | 开始采集的迭代数，默认值0 |     Y     |      Y      |
| record-shapes | action | 是否采集算子的InputShapes和InputTypes，设置参数采集，默认不采集 |     Y     |      N      |
| profile-memory | action | 是否采集算子内存信息，设置参数采集，默认不采集 |     Y     |      Y      |
| with-stack | action | 是否采集Python调用栈，设置参数采集，默认不采集，当前MindSpore和PyTorch框架都支持 |     Y     |      Y      |
| with-flops | action | 是否采集算子flops，设置参数采集，默认不采集 |     Y     |      N      |
| with-modules | action | 是否采集modules层级的Python调用栈，设置参数采集，默认不采集 |     Y     |      N      |
| analyse | action | 采集后是否自动解析，设置参数解析，默认不解析 |     Y     |      Y      |
| l2-cache | action | 是否采集L2 Cache数据，设置参数采集，默认不采集 |     Y     |      Y      |
| op-attr | action | 是否采集算子属性信息，设置参数采集，默认不采集 |     Y     |      N      |
| msprof-tx | action | 是否使能MSTX，设置参数采集，默认使能 |     Y     |      Y      |
| data-simplification | String | 解析完成后是否数据精简，可选值范围[`true`, `false`]，默认值`true` |     Y     |      Y      |
| activities | String | 控制CPU、NPU事件采集范围，可选值范围[`CPU,NPU`, `NPU,CPU`, `CPU`, `NPU`]，默认值`CPU,NPU` |     Y     |      Y      |
| profiler-level | String | 控制profiler的采集等级，可选值范围[`Level_none`, `Level0`, `Level1`, `Level2`]，默认值`Level0` |     Y     |      Y      |
| aic-metrics | String | AI Core的性能指标采集项，可选值范围[`AiCoreNone`, `PipeUtilization`, `ArithmeticUtilization`, `Memory`, `MemoryL0`, `ResourceConflictRatio`, `MemoryUB`, `L2Cache`, `MemoryAccess`]，默认值`AiCoreNone` |     Y     |      Y      |
| export-type | String | profiler解析导出数据的类型，可选值范围[`Text`, `Db`]，默认值`Text` |     Y     |      Y      |
| gc-detect-threshold | Option<f32> | GC检测阈值，单位ms，只采集超过阈值的GC事件。该参数为可选参数，默认不设置时不开启GC检测 |     Y     |      N      |


- nputrace使用方法

Step0: 参考`3.编译`章节完成dynolog的编译，以及dynolog_npu_plugin wheel包的编译和安装。

Step1：拉起dynolog daemon进程
```bash
# 方法1和方法2 二选一
# 方法1：使用systemd拉起service
# 修改配置文件/etc/dynolog.gflags, 使能ipc_monitor
echo "--enable_ipc_monitor" | sudo tee -a /etc/dynolog.gflags
sudo systemctl start dynolog

# 方法2：命令行执行
dynolog --enable-ipc-monitor

#dynolog daemon的日志路径为：/var/log/dynolog.log
```

Step 2：使能dynolog trace dump环境变量
```bash
export KINETO_USE_DAEMON=1
```

Step 3: 拉起训练任务
```bash
# 训练任务中需要使用pytorch的优化器/继承原生优化器
bash train.sh
```

Step 4：使用dyno CLI动态触发trace dump
```bash
# 示例1：从第10个step开始采集，采集2个step，采集框架、CANN和device数据，同时采集完后自动解析以及解析完成不做数据精简，落盘路径为/tmp/profile_data
dyno nputrace --start-step 10 --iterations 2 --activities CPU,NPU --analyse --data-simplification false --log-file /tmp/profile_data

# 示例2：从第10个step开始采集，采集2个step，只采集CANN和device数据，同时采集完后自动解析以及解析完成后开启数据精简，落盘路径为/tmp/profile_data
dyno nputrace --start-step 10 --iterations 2 --activities NPU --analyse --data-simplification true --log-file /tmp/profile_data

# 示例3：从第10个step开始采集，采集2个step，只采集CANN和device数据，只采集不解析，落盘路径为/tmp/profile_data
dyno nputrace --start-step 10 --iterations 2 --activities NPU --log-file /tmp/profile_data

# 示例4：多机场景下向特定机器x.x.x.x发送参数信息，参数表示从第10个step开始采集，采集2个step，只采集CANN和device数据，只采集不解析，落盘路径为/tmp/profile_data
dyno --hostname x.x.x.x nputrace --start-step 10 --iterations 2 --activities NPU --log-file /tmp/profile_data
```

### NPU Monitor功能（该功能当前只支持命令行参数，完整功能请使用[poc](https://gitee.com/ascend/mstt/tree/poc/msmonitor)分支）
NPU Monitor基于MSPTI/MSTX能力开发，实现了轻量级在线监控能力，能够用于性能问题的初步定位。

```bash
dyno npu-monitor --help
```

- npu-monitor使用方式

```bash
dyno npu-monitor [SUBCOMMANDS]
```

npu-monitor子命令支持的参数选项

| 子命令 | 参数类型 | 说明 | PyTorch支持 | MindSpore支持 |
|-------|-------|-------|:---------:|:-----------:|
| npu-monitor-start | action | 开启性能监控，设置参数开启，默认不采集 | Y | Y |
| npu-monitor-stop | action | 停止性能监控，设置参数开启，默认不采集 | Y | Y |
| report-interval-s | int | 性能监控数据上报周期，单位s，需要在启动时设置。默认值60持 | Y | Y |
| mspti-activity-kind | String | 性能监控数据上报数据类型，可以设置单个或多个，多个类型以逗号分隔，每次设置时刷新全局上报类型。可选值范围[`Marker`, `Kernel`, `API`, `Hccl`, `Memory`, `MemSet`, `MemCpy`] , 默认值`Marker` | Y | Y |

- npu-monitor使用方法

Step1： 拉起dynolog daemon进程
```bash
# 方法1和方法2 二选一
# 方法1：使用systemd拉起service
# 修改配置文件/etc/dynolog.gflags, 使能ipc_monitor
echo "--enable_ipc_monitor" | sudo tee -a /etc/dynolog.gflags
sudo systemctl start dynolog

# 方法2：命令行执行
dynolog --enable-ipc-monitor

# 使用Prometheus上报数据需要指定参数：--use_prometheus
# dynolog daemon的日志路径为：/var/log/dynolog.log
```

Step 2：使能dynolog环境变量
```bash
export KINETO_USE_DAEMON=1
```

Step 3: 拉起训练任务
```bash
# 训练任务拉起前需要设置LD_PRELOAD
# 示例：export LD_PRELOAD=/usr/local/ascend-tookit/latest/lib64/libmspti.so
export LD_PRELOAD=<CANN tookkit安装路径>/ascend-tookit/latest/lib64/libmspti.so

# 训练任务中需要使用pytorch的优化器/继承原生优化器
bash train.sh
```

Step 4：使用dyno CLI使能npu-monitor
```bash
# 示例1：开启性能监控，使用默认配置
dyno npu-monitor --npu-monitor-start

# 示例2：暂停性能监控
dyno npu-monitor --npu-monitor-stop

# 示例3：性能监控过程中修改配置
# 上报周期30s, 上报数据类型Marker和Kernel
dyno npu-monitor --report-interval-s 30 --mspti-activity-kind Marker,Kernel

# 示例4：性能监控开启时修改配置
# 上报周期30s, 上报数据类型Marker和Kernel
dyno npu-monitor --npu-monitor-start --report-interval-s 30 --mspti-activity-kind Marker,Kernel

# 示例5：多机场景下性能监控开启时修改配置
# 多机场景下向特定机器x.x.x.x发送参数信息，参数表示上报周期30s, 上报数据类型Marker和Kernel
dyno --hostname x.x.x.x npu-monitor --npu-monitor-start --report-interval-s 30 --mspti-activity-kind Marker,Kernel
```

Step5: 观测Prometheus上报数据
```
# Prometheus默认端口为8080
curl 127.0.0.1:8080/metrics
```