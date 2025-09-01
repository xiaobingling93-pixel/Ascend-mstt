# nputrace特性

nputrace通过dyno CLI中的nputrace子命令开启：

```bash
dyno --certs-dir <CERT_DIR> nputrace [SUBCOMMANDS]
```
**说明**：
- 1. dyno和dynolog中--certs-dir传入参数值须保持一致；
- 2. <CERT_DIR>可传入证书路径，如果不使用TLS证书密钥，设置为NO_CERTS。

查看nputrace支持的命令和帮助

```bash
dyno nputrace --help
```

nputrace的SUBCOMMANDS（子命令）选项如下：

| 子命令                   | 参数类型 | 说明                                                                                                                                                                                                                                                                                 | PyTorch支持 | MindSpore支持 |  是否必选  |
|-----------------------|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------:|:-----------:|:----------:|
| --job-id              | u64 | 采集任务的job id，默认值0，dynolog原生参数                                                                                                                                                                                                                                                       |     N     |      N      |  N |
| --pids                | String | 采集任务的pid列表，多个pid用逗号分隔，默认值0，dynolog原生参数                                                                                                                                                                                                                                             |     N     |      N      |  N |
| --process-limit       | u64 | 最大采集进程的数量，默认值3，dynolog原生参数                                                                                                                                                                                                                                                         |     N     |      N      | N |
| --profile-start-time  | u64 | 用于同步采集的Unix时间戳，单位毫秒，默认值0，dynolog原生参数                                                                                                                                                                                                                                               |     N     |      N      | N |
| --duration-ms         | u64 | 采集的周期，单位毫秒，默认值500，dynolog原生参数                                                                                                                                                                                                                                                      |     N     |      N      | N |
| --iterations          | i64 | 采集总迭代数，dynolog原生参数，需与start-step参数同时指定                                                                                                                                                                                                                                        |     Y     |      Y      |  Y |
| --log-file            | String | 采集落盘的路径                                                                                                                                                                                                                                                                            |     Y     |      Y      |  Y |
| --start-step          | i64 | 开始采集的迭代数，设置为-1时表示从下一个step开始采集                                                                                                                                                                                                                                                 |     Y     |      Y      | Y  |
| --record-shapes       | action | 是否采集算子的InputShapes和InputTypes，设置参数采集，默认不采集                                                                                                                                                                                                                                         |     Y     |      Y      |  N |
| --profile-memory      | action | 是否采集算子内存信息，设置参数采集，默认不采集                                                                                                                                                                                                                                                            |     Y     |      Y      | N |
| --with-stack          | action | 是否采集Python调用栈，设置参数采集，默认不采集                                                                                                                                                                                                                                                         |     Y     |      Y      | N |
| --with-flops          | action | 是否采集算子flops，设置参数采集，默认不采集                                                                                                                                                                                                                                                           |     Y     |      N      | N |
| --with-modules        | action | 是否采集modules层级的Python调用栈，设置参数采集，默认不采集                                                                                                                                                                                                                                               |     Y     |      N      | N |
| --analyse             | action | 采集后是否自动解析，设置参数解析，默认不解析                                                                                                                                                                                                                                                             |     Y     |      Y      | N |
| --l2-cache            | action | 是否采集L2 Cache数据，设置参数采集，默认不采集                                                                                                                                                                                                                                                        |     Y     |      Y      | N |
| --op-attr             | action | 是否采集算子属性信息，设置参数采集，默认不采集                                                                                                                                                                                                                                                            |     Y     |      N      | N |
| --msprof-tx           | action | 是否使能MSTX，设置参数采集，默认不使能                                                                                                                                                                                                                                                              |     Y     |      Y      | N |
| --mstx-domain-include | Option<String> | 使能--msprof-tx采集mstx打点数据的情况下，配置该开关，设置实际采集的domain范围，与--mstx-domain-exclude参数互斥，若同时设置，则只有--mstx-domain-include生效。该参数为可选参数，默认不使能。可配置一个或多个domain，例如：--mstx-domain-include domain1, domain2                                                                                              |     Y     |      Y      |  N |
| --mstx-domain-exclude | Option<String> | 使能--msprof-tx采集mstx打点数据的情况下，配置该开关，设置实际不采集的domain范围，与--mstx-domain-include参数互斥，若同时设置，则只有--mstx-domain-include生效。该参数为可选参数，默认不使能。可配置一个或多个domain，例如：--mstx-domain-exclude domain1, domain2                                                                                             |     Y     |      Y      | N |
| --data-simplification | String | 解析完成后是否数据精简，可选值范围[`true`, `false`]，默认值`true`                                                                                                                                                                                                                                       |     Y     |      Y      | N |
| --activities          | String | 控制CPU、NPU事件采集范围，可以设置单个或多个，多个类型以逗号分隔，可选值范围[`CPU`, `NPU`]，默认值`CPU,NPU`                                                                                                                                                                                                               |     Y     |      Y      | N |
| --profiler-level      | String | 控制profiler的采集等级，可选值范围[`Level_none`, `Level0`, `Level1`, `Level2`]，默认值`Level0`                                                                                                                                                                                                      |     Y     |      Y      | N |
| --aic-metrics         | String | AI Core的性能指标采集项，可选值范围[`AiCoreNone`, `PipeUtilization`, `ArithmeticUtilization`, `Memory`, `MemoryL0`, `ResourceConflictRatio`, `MemoryUB`, `L2Cache`, `MemoryAccess`]，当--profiler-level设置为Level_none或Level0，默认值`AiCoreNone`，当--profiler-level设置为Level1或Level2，默认值`PipeUtilization` |     Y     |      Y      | N |
| --export-type         | String | profiler解析导出数据的类型，只能设置其中一个，不能同时设置db和text，可选值范围[`Text`, `Db`]，默认值`Text`                                                                                                                                                                                                             |     Y     |      Y      | N |
| --gc-detect-threshold | Option<f32> | GC检测阈值，单位ms，只采集超过阈值的GC事件。默认不设置时不开启GC检测                                                                                                                                                                                                                                             |     Y     |      N      | N |
| --host-sys            | String | 采集[host侧系统数据](https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/T&ITools/Profiling/atlasprofiling_16_0014.html)(CPU利用率、内存利用率、磁盘I/O利用率、网络I/O利用率等)。可以设置单个或多个，多个类型以逗号分隔，可选值范围[`cpu`, `mem`, `disk`, `network`, `osrt`] , 默认不设置时不开启host侧系统数据采集                              |     Y     |      Y      | N |
| --sys-io              | action | 采集NIC、ROCE数据。设置参数采集，默认不采集                                                                                                                                                                                                                                                          |     Y     |      Y      | N |
| --sys-interconnection | action | 采集集合通信带宽数据（HCCS）、PCIe、片间传输带宽数据。设置参数采集，默认不采集                                                                                                                                                                                                                                        |     Y     |      Y      | N |

## nputrace使用方法

Step 1：拉起dynolog daemon进程，详细介绍请参考[dynolog介绍](./dynolog.md)

- 示例
```bash
# 命令行方式开启dynolog daemon
dynolog --enable-ipc-monitor --certs-dir /home/server_certs
```

Step 2：在训练/推理任务拉起窗口使能dynolog环境变量
```bash
export MSMONITOR_USE_DAEMON=1
```

Step 3：拉起训练/推理任务
```bash
# 训练任务中需要使用pytorch的优化器/继承原生优化器
bash train.sh
```

Step 4：使用dyno CLI动态触发trace dump
```bash
# 示例1：从第10个step开始采集，采集2个step，采集框架、CANN和device数据，同时采集完后自动解析以及解析完成不做数据精简，落盘路径为/tmp/profile_data
dyno --certs-dir /home/client_certs nputrace --start-step 10 --iterations 2 --activities CPU,NPU --analyse --data-simplification false --log-file /tmp/profile_data

# 示例2：从下一个step开始采集，采集2个step，采集框架、CANN和device数据，同时采集完后自动解析以及解析完成不做数据精简，落盘路径为/tmp/profile_data
dyno --certs-dir /home/client_certs nputrace --start-step -1 --iterations 2 --activities CPU,NPU --analyse --data-simplification false --log-file /tmp/profile_data

# 示例3：从第10个step开始采集，采集2个step，只采集CANN和device数据，同时采集完后自动解析以及解析完成后开启数据精简，落盘路径为/tmp/profile_data
dyno --certs-dir /home/client_certs nputrace --start-step 10 --iterations 2 --activities NPU --analyse --data-simplification true --log-file /tmp/profile_data

# 示例4：从第10个step开始采集，采集2个step，只采集CANN和device数据，只采集不解析，落盘路径为/tmp/profile_data

dyno --certs-dir /home/client_certs nputrace --start-step 10 --iterations 2 --activities NPU --log-file /tmp/profile_data

# 示例5：多机场景下向特定机器x.x.x.x发送参数信息，参数表示从第10个step开始采集，采集2个step，只采集CANN和device数据，只采集不解析，落盘路径为/tmp/profile_data
dyno --certs-dir /home/client_certs --hostname x.x.x.x nputrace --start-step 10 --iterations 2 --activities NPU --log-file /tmp/profile_data
```
nputrace落盘的数据格式和交付件介绍请参考[MindSpore&PyTorch框架性能数据文件参考](https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/T&ITools/Profiling/atlasprofiling_16_0177.html)