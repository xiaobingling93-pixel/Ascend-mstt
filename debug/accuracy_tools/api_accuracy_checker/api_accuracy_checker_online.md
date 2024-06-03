# 在线精度预检

为了应对大模型场景下，通过离线预检方式dump API输入输出数据导致的存储资源紧张问题，提供在线精度预检功能。本功能实现在执行NPU训练操作的过程中，通过TCP/IP协议在NPU Host与GPU Host设备间建立连接，将NPU上对应API的输入数据在GPU设备上运行，将两份输出数据进行比对，得到预检比对结果，从而减少数据dump的步骤，降低存储资源的占用。

## 在线精度预检流程

在线精度预检操作流程如下：

1. 准备GPU和NPU可正常运行的训练环境，PyTorch版本大于等于2.0，并保证两台Host在同一局域网内可正常通信。
2. GPU和NPU Host设备上同时安装预检工具，详见《[Ascend模型精度预检工具](./README.md)》，其中在线预检要多安装twisted依赖，该依赖为Python模块。
3. 分别配置GPU侧、NPU侧的config.yaml文件。
4. 在GPU侧运行run_ut.py。
5. 在NPU侧配置训练脚本。
6. 在NPU侧执行训练。

## 在线精度预检操作指导

### 配置config.yaml文件

安装完成预检工具后，需要分别在GPU和NPU环境下配置config.yaml文件。其中需要重点关注文件中的is_online、is_benchmark_device、host和port参数的配置，保障在线预检时GPU和NPU两台设备间的通信正常。

文件路径为：att/debug/accuracy_tools/api_accuracy_checker/config.yaml

| 参数名称            | 说明                                                         | 是否必选 |
| ------------------- | ------------------------------------------------------------ | -------- |
| dump_path           | 设置在线预检结果输出路径，str类型，默认与run_ut.py同目录。若指定目录不存在，则自动创建。在GPU侧配置生效。 | 否       |
| real_data           | 真实数据模式，bool类型，可取值True或False，默认为False，表示随机数据模式，配置为True后开启真实数据模式。在线预检模式下该参数不生效。 | 否       |
| enable_dataloader   | 自动dump数据开关，bool类型，可取值True（开启）、False（关闭），默认关闭。在线预检模式下该参数不生效。 | 否       |
| target_iter         | 指定dump某个step的数据，list[int]类型，默认为[1]，须指定为训练脚本中存在的step。target_iter为list格式，可配置逐个step，例如：target_iter=[0,1,2]；也可以配置step范围，例如：target_iter=list(range(0,9))，表示dump第0到第8个step。在线预检模式下，GPU侧和NPU侧的配置值需相同，所有step的预检结果会被写入到同一个Rank的结果文件中。 | 否       |
| white_list          | API dump白名单，指定dump具体API数据，也可以直接配置预检的API白名单，list[str]类型。参数示例：white_list=["conv1d", "conv2d"]。默认未配置白名单，即dump全量API数据。在线预检模式下，在GPU侧配置生效。 | 否       |
| error_data_path     | 配置保存精度未达标的API输入输出数据路径，str类型。在线预检模式下该参数不生效。 | 否       |
| precision           | 浮点数表示位数，int类型，默认取小数点后14位。                | 否       |
| is_online           | 在线预检模式开关，bool类型，可取值True（开启）、False（关闭），默认关闭。 | 是       |
| is_benchmark_device | 在线预检模式标杆设备，bool类型，可取值True（开启）、False（关闭），默认关闭。GPU侧配置开启，表示为标杆设备；NPU侧配置关闭，表示为待比对设备。 | 是       |
| host                | 在线预检模式信息接收端IP，str类型，用于GPU设备和NPU设备间进行通信，GPU侧配置为本机地址127.0.0.1或本机局域网IP，NPU侧须配置为GPU侧的局域网IP地址。 | 是       |
| port                | 在线预检模式信息接收端端口号，int类型，用于GPU设备和NPU设备间进行通信，GPU侧配置为本机可用端口，NPU侧须配置为GPU侧的端口号。 | 是       |
| rank_list           | 指定在线预检的Rank ID，默认值为[0]，list[int]类型，应配置为大于等于0的整数，且须根据实际卡的Rank ID配置，若所配置的值大于实际训练所运行的卡的Rank ID，则在线预检输出数据为空。GPU和NPU须配置一致。 | 否       |

### 在GPU侧运行run_ut.py

由于GPU侧为通信接收端，需先于NPU侧执行run_ut操作。

GPU侧配置好config.yaml文件后执行run_ut.py脚本，此时GPU处于预检等待状态，当NPU侧启动训练后将预检的API输入和输出数据发送到GPU侧时，GPU启动预检操作。

命令如下：

```bash
cd $ATT_HOME/debug/accuracy_tools/api_accuracy_checker/run_ut
python run_ut.py
```

### 在NPU侧配置训练脚本

在NPU训练脚本中添加如下代码以获取run_ut操作的预检API输入和输出数据：

```python
from api_accuracy_checker.dump import dump as DP
from api_accuracy_checker.dump import dispatch

...

DP.dump.start()    

...

DP.dump.stop()    
DP.dump.step()    # 在DP.dump.stop()后加入DP.dump.step()即可确定需要预检的step
```

### 在NPU侧执行训练脚本

配置完NPU侧训练脚本后即可执行训练脚本，命令示例如下：

```bash
bash train.sh
```

训练脚本执行完毕后，在GPU侧dump_path目录下生成比对结果文件，详细介绍请参见《[Ascend模型精度预检工具](./README.md)》中的”**预检结果**“。

