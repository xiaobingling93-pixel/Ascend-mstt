# npumonitor特性

npumonitor通过dyno CLI中的npumonitor子命令开启：

```bash
dyno --certs-dir <CERT_DIR> npu-monitor [SUBCOMMANDS]
```

查看npumonitor支持的命令和帮助

```bash
dyno npumonitor --help
```

npu-monitor的SUBCOMMANDS（子命令）选项如下：

| 子命令                   | 参数类型 | 说明                                                                                                                                                  | PyTorch支持 | MindSpore支持 |    是否必选     |
|-----------------------|-------|------------------------------------------------------------------------------------------------------------------------------------------------------|:---------:|:-----------:|:-----------:|
| --npu-monitor-start   | action | 开启性能监控，设置参数后生效，默认不生效                                                                                                                                | Y | Y | N |
| --npu-monitor-stop    | action | 停止性能监控，设置参数后生效，默认不生效                                                                                                                                | Y | Y | N |
| --report-interval-s   | int | 性能监控数据上报周期，单位s，需要在启动时设置。默认值60                                                                                                                       | Y | Y | N |
| --mspti-activity-kind | String | 性能监控数据上报数据类型，可以设置单个或多个，多个类型以逗号分隔，每次设置时刷新全局上报类型。可选值范围[`Marker`, `Kernel`, `API`, `Hccl`, `Memory`, `MemSet`, `MemCpy`, `Communication`] , 默认值`Marker` | Y | Y | N |

## npu-monitor使用方法

Step 1： 拉起dynolog daemon进程
```bash
# 方法1和方法2 二选一
# 方法1：使用systemd拉起service
# 修改配置文件/etc/dynolog.gflags, 使能ipc_monitor
echo "--enable_ipc_monitor" | sudo tee -a /etc/dynolog.gflags
sudo systemctl start dynolog

# 方法2：命令行执行
dynolog --enable-ipc-monitor --certs-dir /home/server_certs

# 使用Tensorboard上报数据需要指定参数：--metric_log_dir, 指定Tensorboard文件落盘文件
# dynolog daemon的日志路径为：/var/log/dynolog.log
```

Step 2：在训练任务拉起窗口使能dynolog环境变量
```bash
export MSMONITOR_USE_DAEMON=1
```

Step 3: 配置Msmonitor日志路径(可选，默认路径为当前目录下的msmonitor_log)
```bash
export MSMONITOR_LOG_PATH=<LOG PATH>
# 示例：
export MSMONITOR_LOG_PATH=/tmp/msmonitor_log
```

Step 4: 拉起训练任务
```bash
# 训练任务拉起前需要设置LD_PRELOAD
# 示例：export LD_PRELOAD=/usr/local/Ascend/ascend-toolkit/latest/lib64/libmspti.so
export LD_PRELOAD=<CANN toolkit安装路径>/ascend-toolkit/latest/lib64/libmspti.so

# 训练任务中需要使用pytorch的优化器/继承原生优化器
bash train.sh
```

Step 5：使用dyno CLI使能npu-monitor
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

# 示例5：多机场景下性能监控开启时修改配置
# 多机场景下向特定机器x.x.x.x发送参数信息，参数表示上报周期30s, 上报数据类型Marker和Kernel
dyno --certs-dir /home/client_certs --hostname x.x.x.x npu-monitor --npu-monitor-start --report-interval-s 30 --mspti-activity-kind Marker,Kernel
```

Step6: 观测Tensorboard上报数据
```
# Tensorboard存储数据路径在指定参数metric_log_dir下
# 请确保安装了Tensorboard：

pip install tensorboard

# 然后运行：
# metric_log_dir为启动守护进程时所指定参数
tensorboard --logdir={metric_log_dir} 

# 打开浏览器访问http://localhost:6006即可看到对应可视化图表, 其中6006为tensorboard默认端口
```
> tensorboard 具体使用参数见https://github.com/tensorflow/tensorboard