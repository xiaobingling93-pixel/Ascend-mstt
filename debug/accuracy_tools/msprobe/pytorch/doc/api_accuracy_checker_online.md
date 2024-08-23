# 在线精度预检

为了应对大模型场景下，通过离线预检方式dump API输入输出数据导致的存储资源紧张问题，提供在线精度预检功能。本功能实现在执行NPU训练操作的过程中，通过TCP/IP协议在NPU
Host与GPU Host设备间建立连接，将NPU上对应API的输入数据在GPU设备上运行，将两份输出数据进行比对，得到预检比对结果，从而减少数据dump的步骤，降低存储资源的占用。
针对偏差较大的算子，两方比对（NPUvsGPU）的方法缺少裁判进行裁定。 参考离线预检，在线预检场景同时支持两方比对和三方比对方式，按照api的精度标准要求，选择比对两方比对和三方比对。

## 在线精度预检流程

在线精度预检当前支持**局域网场景**和**共享存储场景**，请根据不同的场景选择对应的配置。

在线精度预检操作流程如下：

1. 准备GPU和NPU可正常运行的训练环境，PyTorch版本大于等于2.0，并保证两台Host在同一局域网内可正常通信或能通过共享存储进行通信。
2. GPU和NPU Host设备上同时安装msprobe工具，详见《[MindStudio精度调试工具](./../../README.md)
   》，其中在线预检要多安装twisted、pyOpenSSL依赖，该依赖为Python模块。
3. 分别配置GPU侧、NPU侧的config.json文件。
4. 在GPU侧运行msprobe -f pytorch run_ut -config ./config.json。
5. 在NPU侧配置训练脚本。
6. 在NPU侧执行训练。

## 在线精度预检操作指导

### 配置config.json文件

安装完成预检工具后，需要分别在GPU和NPU环境下分别配置config.json。其中需要重点关注文件中的is_online、is_benchmark_device、host和port参数的配置，保障在线预检时GPU和NPU两台设备间的通信正常。

#### GPU侧在线预检配置说明

| 参数名称            | 说明                                                                                                                    | 是否必选 |
|-----------------|-----------------------------------------------------------------------------------------------------------------------|------|
| task            | 任务名称，str类型，配置为run_ut表示预检任务。通过其他字段is_online判断离线预检、在线预检任务。                                                              | 是    |
| white_list      | 预检的API白名单，list[str]类型。参数示例：white_list=["conv1d", "conv2d"]。默认未配置白名单，即预检全量API数据。                                       | 否    |
| black_list      | 预检的API黑名单，list[str]类型。参数示例：white_list=["conv1d", "conv2d"]。默认未配置黑名单，即预检全量API数据。                                       | 否    |
| error_data_path | 配置保存精度未达标的API输入输出数据路径，str类型。在线预检模式下该参数不生效。                                                                            | 否    |
| is_online       | 在线预检模式开关，bool类型，可取值True（开启）、False（关闭），默认关闭。                                                                           | 是    |
| nfs_path        | 在线预检模式共享存储目录路径，str类型，用于GPU设备和NPU设备间进行通信。配置该参数后host、port和tls_path不生效。                                                  | 否    |
| host            | 在线预检模式局域网场景信息接收端IP，str类型，用于GPU设备和NPU设备间进行通信，GPU侧配置为本机地址127.0.0.1或本机局域网IP。局域网场景时，不能配置nfs_path参数，否则局域网场景不生效。            | 否    |
| port            | 在线预检模式局域网场景信息接收端端口号，int类型，用于GPU设备和NPU设备间进行通信，GPU侧配置为本机可用端口。局域网场景时，不能配置nfs_path参数，否则局域网场景不生效。                          | 否    |
| rank_list       | 指定在线预检的Rank ID，默认值为[0]，list[int]类型，应配置为大于等于0的整数，且须根据实际卡的Rank ID配置，若所配置的值大于实际训练所运行的卡的Rank ID，则在线预检输出数据为空。GPU和NPU须配置一致。 | 是    |
| tls_path        | 在线预检模式局域网场景SSL证书路径，该路径下包含私钥文件server.key和公钥文件server.crt，str类型，默认值为空，表示明文传输api数据，否则采用TLS1.2加密传输，加密传输时安全性较高，传输速率较低       | 否    |

#### NPU侧在线预检配置说明

| 参数名称             | 说明                                                                                                     | 是否必选 |
|------------------|--------------------------------------------------------------------------------------------------------|------|
| task             | 任务名称，str类型，配置为tensor表示dump API统计信息和完全复刻整网的API运行情况的真实数据。通过字段onlin_run_ut判断是否使用在线预检功能。                   | 是    |
| dump_path        | dump路径，str类型，配置为合法路径即可，兼容tensor任务静态检查                                                                  | 是    |
| level            | dump级别，str类型，在线预检时配置为L1，表示dump API级精度数据。在线预检可不配置，默认取值L1。                                               | 是    |
| rank             | 指定对某张卡上的数据进行dump，list[int]类型，默认未配置(表示dump所有卡的数据)，需要与GPU侧配置项rank_list保持一致。                              | 否    |
| step             | 指定dump某个step的数据，list[int]类型，默认未配置，表示dump所有step的数据。dump特定step时，须指定为训练脚本中存在的step。                        | 否    |
| seed             | 随机种子数，int类型，默认值为1234。通过固定随机数保证模型的输入或输出一致。                                                              | 否    |
| is_deterministic | 确定性计算模式，bool类型，可取值true（开启）或false（关闭），默认关闭。                                                             | 否    |
| scope            | dump范围，list[str]类型，默认未配置（list也未配置时师表dump所有api的额数据），配置方式参考[《config配置文件说明》](./../../config/README.md)    | 否    |
| list             | 自定义dump范围，list[str]类型，默认未配置（scope也未配置时表示dump所有api的数据），配置方式参考[《config配置文件说明》](./../../config/README.md) | 否    |
| online_run_ut    | 在线预检模式开关，bool类型，可取值True（开启）、False（关闭），默认关闭。                                                            | 是    |
| nfs_path         | 在线预检模式共享存储目录路径，str类型，用于GPU设备和NPU设备间进行通信。配置该参数后host、port和tls_path不生效。                                            | 否    |
| host             | 在线预检模式局域网场景信息接收端IP，str类型，用于GPU设备和NPU设备间进行通信，NPU侧须配置为GPU侧的局域网IP地址。局域网场景时，不能配置nfs_path参数，否则局域网场景不生效。     | 否    |
| port             | 在线预检模式局域网场景SSL证书路径，该路径下包含私钥文件client.key和公钥文件client.crt，str类型，默认值为空，表示明文传输api数据，否则采用TLS1.2加密传输，加密传输时安全性较高，传输速率较低        | 否    |

#### 局域网场景配置示例

若采用TLS1.2协议加密传输api数据，需配置SSL证书，可参考如下生成自签名证书方法，仅供调试使用，生产环境请申请正式证书。
```shell
创建私钥文件server.key
openssl genrsa -out server.key 2048

创建签名请求文件server.csr，如无要求，全部回车
openssl req -new -key server.key -out server.csr

自签名，生成1年期公钥文件server.crt
openssl x509 -req -days 365 -in server.csr -signkey server.key -out server.crt
```

GPU侧：

```json
{
  "task": "run_ut",
  "run_ut": {
    "white_list": [],
    "black_list": [],
    "error_data_path": "./",
    "is_online": true,
    "nfs_path": "",
    "host": "127.0.0.1",
    "port": 59208,
    "rank_list": [0]
  }
}
```

NPU侧：

```json
{
  "task": "tensor",
  "dump_path": "./dump_path",
  "rank": [0],
  "step": [0],
  "level": "L1",
  "seed": 1234,
  "is_deterministic": true,
  "tensor": {
    "scope": [],
    "list": [],
    "online_run_ut": true,
    "nfs_path": "",
    "host": "xx.xx.xx.x",
    "port": 59208
  }
}
```

#### 共享存储场景配置示例

若复制下列示例，请删除注释后使用。

GPU侧：

```json
{
  "task": "run_ut",
  "run_ut": {
    "white_list": [],
    "black_list": [],
    "error_data_path": "./",
    "is_online": true,
    "nfs_path": "/nfs/xxx/data",
    "host": "",
    "port": -1,
    "rank_list": [0]
  }
}
```

NPU侧：

```json
{
  "task": "tensor",
  "dump_path": "./dump_path",
  "rank": [0],
  "step": [0],
  "level": "L1",
  "seed": 1234,
  "is_deterministic": true,
  "tensor": {
    "scope": [],
    "list": [],
    "online_run_ut": true,
    "nfs_path": "/nfs/xxx/data",
    "host": "",
    "port": -1
  }
}
```

### 在GPU侧运行run_ut

由于GPU侧为通信接收端，需先于NPU侧执行run_ut操作，命令如下：

```bash
msprobe -f pytorch run_ut -config ./config.json
```

GPU侧配置好config.json文件后执行run_ut命令，此时GPU处于预检等待状态：

- 局域网场景：当NPU侧启动训练后将预检的API输入和输出数据发送到GPU侧时，GPU启动预检操作。
- 共享存储场景：当NPU侧启动训练后将预检的API输入和输出数据发送到共享存储时，GPU启动预检操作。

### 在NPU侧配置训练脚本

在NPU训练脚本中添加如下代码以获取run_ut操作的预检API输入和输出数据：

```python
from msprobe.pytorch import PrecisionDebugger

debugger = PrecisionDebugger("config.json")
...

debugger.start()

...

debugger.stop()
debugger.step()
```

### 在NPU侧执行训练脚本

配置完NPU侧训练脚本后即可执行训练脚本，命令示例如下：

```bash
bash train.sh
```

训练脚本执行完毕后，在GPU侧dump_path目录下生成比对结果文件，
`accuracy_checking_result_{timestamp}_rank{rank_id}.csv`和`accuracy_checking_details_{timestamp}_rank{rank_id}.csv`记录两方比对结果
`api_precision_compare_result_{timestamp}_rank{rank_id}.csv`和`api_precision_compare_details_{timestamp}_rank{rank_id}.csv`记录三方比对结果。
详细介绍请参见《[精度预检工具](./api_accuracy_checker.md)》中的“**预检结果**”。