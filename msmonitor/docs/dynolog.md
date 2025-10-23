# dynolog介绍

dynolog负责接收dyno CLI的RPC请求，触发nputrace和npumonitor功能。

- dynolog daemon可以通过systemd或者命令行任意一种方法开启

```bash
# 方法1：使用systemd拉起service
# 修改配置文件/etc/dynolog.gflags, 使能ipc_monitor
echo "--enable_ipc_monitor" | sudo tee -a /etc/dynolog.gflags
sudo systemctl start dynolog
```

```bash
# 方法2：命令行执行
dynolog --enable-ipc-monitor --certs-dir /home/server_certs
```

## dynolog常用参数

| 命令                  | 参数类型   | 说明                                                    | 是否必选 |
|---------------------|--------|-------------------------------------------------------|:----:|
| --enable-ipc-monitor | action | 是否启用IPC监控功能，用于与dyno进行通信，设置参数开启，默认不开启                  |  N   |
| --port              |  i32   | dynolog daemon进程监听的端口号，默认值1778                        |  N   |
| --certs-dir         | String | 用于指定dyno与dynolog RPC通信时TLS证书的路径，当值为`NO_CERTS`时不使用证书校验 |  Y   |
| --metric_log_dir    | String | 用于指定metric数据的落盘路径                                     |  N   |
| --use_JSON          | action | 是否使用JSON格式记录metric数据到日志中，默认不启用                        |  N   |

