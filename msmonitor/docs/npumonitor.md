# npu-monitor特性

npu-monitor通过dyno CLI中的npu-monitor子命令开启：

```bash
dyno --certs-dir <CERT_DIR> npu-monitor [SUBCOMMANDS]
```
**说明**：
- 1. dyno和dynolog中--certs-dir传入参数值须保持一致；
- 2. <CERT_DIR>可传入证书路径，如果不使用TLS证书密钥，设置为NO_CERTS。


查看npu-monitor支持的命令和帮助

```bash
dyno npu-monitor --help
```

npu-monitor的SUBCOMMANDS（子命令）选项如下：

| 子命令                   | 参数类型 | 说明                                                                                                                                                  | PyTorch支持 | MindSpore支持 |    是否必选     |
|-----------------------|-------|------------------------------------------------------------------------------------------------------------------------------------------------------|:---------:|:-----------:|:-----------:|
| --npu-monitor-start   | action | 开启性能监控，设置参数后生效，默认不生效                                                                                                                                | Y | Y | N |
| --npu-monitor-stop    | action | 停止性能监控，设置参数后生效，默认不生效                                                                                                                                | Y | Y | N |
| --report-interval-s   | int | 性能监控数据上报周期，单位s，需要在启动时设置。默认值60                                                                                                                       | Y | Y | N |
| --mspti-activity-kind | String | 性能监控数据上报数据类型，可以设置单个或多个，多个类型以逗号分隔，每次设置时刷新全局上报类型。可选值范围[`Marker`, `Kernel`, `API`, `Hccl`, `Memory`, `MemSet`, `MemCpy`, `Communication`] , 默认值`Marker` | Y | Y | N |
| --log-file            | String | 性能数据采集落盘的路径，当前仅支持`mspti-activity-kind`设置为`Marker`、`Kernel`、`API`、`Communication`，4种类型数据的导出，落盘数据格式可选为DB、Jsonl（详情参考`export-type`参数说明），默认值为空，表示不落盘 | Y | Y | N |
| --export-type         | String | 性能数据采集落盘的格式，仅在用户设置了`log-file`参数后生效，可选值范围[`DB`, `Jsonl`]，默认值`DB`<br> **1.** 若设置为`DB`，则落盘数据为DB格式，落盘文件名为`msmonitor_{process_id}_{timestamp}_{rank_id}.db`，DB内容说明请参考[msprof导出db格式数据说明](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/devaids/Profiling/atlasprofiling_16_1144.html)，可使用[MindStudio Insight](https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html)工具进行可视化呈现（MindStudio Insight暂不支持呈现单进程多卡场景采集的msmonitor.db数据） <br> **2.** 若设置为`Jsonl`，则落盘数据为Jsonl格式，落盘文件名为`msmonitor_{process_id}_{timestamp}_{rank_id}.jsonl`，Jsonl文件每行包含一条完整的Json格式的性能数据，支持设置以下环境变量对落盘过程进行调节 <br> **MSMONITOR_JSONL_BUFFER_CAPACITY**：设置落盘 RingBuffer 大小，默认值 524288（$2^{19}$），支持的设置范围为 [8192，2097152]（即 [$2^{13}$，$2^{21}$]）。 <br> **MSMONITOR_JSONL_MAX_DUMP_INTERVAL**：设置落盘最长时间间隔（单位：ms），当前时间与上次落盘的间隔超过该阈值时，将自动触发落盘，默认值 30000ms，最小值限制为 1000ms <br> **MSMONITOR_JSONL_ROTATE_LOG_LINES**：设置单个 Jsonl 文件的 Json 数据条数上限，超出该阈值将新建文件落盘。默认值 10000，支持设置范围为 [100, 500000] <br> **MSMONITOR_JSONL_ROTATE_LOG_FILES**：设置单次采集的 Jsonl 文件落盘数量，超出该阈值时将删除最早落盘的文件。默认值 -1（不开启此功能），手动设置时最小值限制为 2 | Y | Y | N |


## npu-monitor使用方法

Step 1： 拉起dynolog daemon进程，详细介绍请参考[dynolog介绍](./dynolog.md)

- 示例
```bash
# 命令行方式开启dynolog daemon
dynolog --enable-ipc-monitor --certs-dir /home/server_certs

# 如需使用Tensorboard展示数据，传入参数--metric_log_dir用于指定Tensorboard文件落盘路径
# 例如：
dynolog --enable-ipc-monitor --certs-dir /home/server_certs --metric_log_dir /tmp/metric_log_dir # dynolog daemon的日志路径为：/var/log/dynolog.log
```

Step 2：在训练/推理任务拉起窗口使能dynolog环境变量
```bash
export MSMONITOR_USE_DAEMON=1
```

Step 3：配置Msmonitor日志路径（可选，默认路径为当前目录下的msmonitor_log）
```bash
export MSMONITOR_LOG_PATH=<LOG PATH>
# 示例：
export MSMONITOR_LOG_PATH=/tmp/msmonitor_log
```

Step 4：设置LD_PRELOAD使能MSPTI
```bash
# 示例：export LD_PRELOAD=/usr/local/Ascend/ascend-toolkit/latest/lib64/libmspti.so
export LD_PRELOAD=<CANN toolkit安装路径>/cann/lib64/libmspti.so
 ```

Step 5：拉起训练/推理任务
```bash
# 训练任务中需要使用pytorch的优化器/继承原生优化器
bash train.sh
```

Step 6：使用dyno CLI使能npu-monitor
```bash
# 示例1：开启性能监控，使用默认配置
dyno --certs-dir /home/client_certs npu-monitor --npu-monitor-start

# 示例2：暂停性能监控
dyno --certs-dir /home/client_certs npu-monitor --npu-monitor-stop

# 示例3：性能监控过程中修改配置
# 上报周期30s, 上报数据类型Marker和Kernel
dyno --certs-dir /home/client_certs npu-monitor --report-interval-s 30 --mspti-activity-kind Marker,Kernel

# 示例4：性能监控开启时修改配置
# 上报周期30s, 上报数据类型Marker和Kernel
dyno --certs-dir /home/client_certs npu-monitor --npu-monitor-start --report-interval-s 30 --mspti-activity-kind Marker,Kernel

# 示例5：性能监控开启时修改配置，开启数据采集落盘
# 数据落盘路径为/tmp/msmonitor_db，落盘周期为30s，采集数据类型为Marker，Kernel，Communication
dyno --certs-dir /home/client_certs npu-monitor --npu-monitor-start --report-interval-s 30 --mspti-activity-kind Marker,Kernel,Communication --log-file /tmp/msmonitor_db

# 示例6：多机场景下性能监控开启时修改配置
# 多机场景下向特定机器x.x.x.x发送参数信息，参数表示上报周期30s, 上报数据类型Marker和Kernel
dyno --certs-dir /home/client_certs --hostname x.x.x.x npu-monitor --npu-monitor-start --report-interval-s 30 --mspti-activity-kind Marker,Kernel
```

Step 7：（可选）观测Tensorboard上报数据
```
# 请确保安装了Tensorboard：
pip install tensorboard

# 然后运行：
tensorboard --logdir={metric_log_dir} # metric_log_dir为Step1中dynolog命令行中--metric_log_dir参数指定的路径

# 打开浏览器访问http://localhost:6006即可看到对应可视化图表, 其中localhost为服务器的ip地址，6006为tensorboard默认端口
```
> tensorboard 具体使用参数见https://github.com/tensorflow/tensorboard