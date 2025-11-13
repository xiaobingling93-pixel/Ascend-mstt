# dyno介绍

dyno负责发送dyno CLI的RPC请求，触发nputrace和npu-monitor功能。


## dyno常用参数

| 命令          | 参数类型   | 说明                                                    | 是否必选 |
|-------------|--------|-------------------------------------------------------|:----:|
| --hostname  | String | dynolog daemon进程的主机名，默认值localhost                     |  N   |
| --port      |  i32   | dynolog daemon进程监听的端口号，默认值1778                        |  N   |
| --certs-dir | String | 用于指定dyno与dynolog RPC通信时TLS证书的路径，当值为`NO_CERTS`时不使用证书校验 |  Y   |
| --help      | action | 用于获取dyno命令的使用帮助，查看所有可用选项和功能说明                                       |  N   |
| --version   | action | 用于查询dyno CLI的版本信息                                     |  N   |

## dyno常用子命令

| 命令        | 说明                                                                  |
|-----------|---------------------------------------------------------------------|
| status    | 查询nputrace或者npu-monitor命令的执行状态                                      |
| nputrace  | 发送nputrace相关消息到dynolog daemon，详情请参考[nputrace](./nputrace.md)        |
| npu-monitor | 发送npu-monitor相关消息到dynolog daemon，详情请参考[npu-monitor](./npumonitor.md) |
| help      | 获取dyno命令的使用帮助，查看所有可用选项和功能说明                                         |
| version   | 查询dynolog daemon的版本信息                                                    |

