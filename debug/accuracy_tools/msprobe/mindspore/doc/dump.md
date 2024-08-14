# **精度数据采集**

msprobe工具主要通过在训练脚本内添加dump接口并启动训练的方式来采集精度数据。

执行dump操作需要安装msprobe工具。详见《[MindStudio精度调试工具](../../README.md)》的“工具安装”章节。

## dump接口介绍

### PrecisionDebugger

**功能说明**

通过加载dump配置文件的方式来确定dump操作的详细配置。

PrecisionDebugger可以在from msprobe.mindspore import PrecisionDebugger之后的位置添加。详细使用可参考“**示例代码**”。

**原型**

```Python
PrecisionDebugger(config_path=None)
```

**参数说明**

| 参数名      | 说明                                                         | 是否必选 |
| ----------- | ------------------------------------------------------------ | -------- |
| config_path | 指定dump配置文件路径，String类型。参数示例："./config.json"。未配置该路径时，默认使用[config.json](../../config)文件的默认配置。config.json文件可以配置更多参数，若需要进行更多场景的精度数据dump，建议配置[config.json](../../config/config.json)文件。config.json文件的配置可参考《[配置文件说明](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/config/README.md)》。 | 否       |

### start函数

**功能说明**

启动函数。

在模型初始化之后的位置添加。需要与stop函数一起添加在for循环内。

**原型**

```Python
debugger.start(model = None)
```

该函数为类函数，可以使用debugger.start(model = None)也可以使用PrecisionDebugger.start(model = None)


**参数说明**

| 参数名      | 说明                                                                                    | 是否必选 |
| ----------- |---------------------------------------------------------------------------------------| -------- |
| model | 指具体的mindspore.nn.Cell，默认未配置，L1级别下传入model可以使能对primitive op的dump，否则无法dump primitive op。 | 否       |


### stop函数

**功能说明**

dump停止函数。

在**start**函数之后的任意位置添加。需要与start函数一起添加在for循环内。若需要dump反向数据，则需要添加在反向计算代码之后。

仅MindSpore动态图场景支持。

**原型**

```Python
debugger.stop()
```

该函数为类函数，可以使用debugger.stop()也可以使用PrecisionDebugger.stop()。

### step函数

**功能说明**

结束标识。

在最后一个**stop**函数后或一个step结束的位置添加。

仅MindSpore动态图场景支持。

**原型**

```Python
debugger.step()
```

该函数为类函数，可以使用debugger.step()也可以使用PrecisionDebugger.step()。

## 示例代码

### MindSpore静态图场景

```Python
from msprobe.mindspore import PrecisionDebugger
debugger = PrecisionDebugger(config_path="./config.json")
# 请勿将以上初始化流程插入到循环代码中
# 下面代码也可以用PrecisionDebugger.start()
debugger.start()
...
```

### MindSpore动态图场景

```Python
import mindspore as ms
from msprobe.mindspore import PrecisionDebugger

# 请勿将PrecisionDebugger的初始化插入到循环代码中
debugger = PrecisionDebugger(config_path="./config.json")

# 模型、损失函数的定义以及初始化等操作
# ...

# 数据集迭代的地方往往是模型开始训练的地方
for data, label in data_loader:
    debugger.start()    # 开启数据dump
    net = Model()
    # 如下是模型每个step执行的逻辑
    grad_net = ms.grad(net)(data)
    # ...
    debugger.stop()     # 关闭数据dump
    debugger.step()     # 结束一个step的dump
```

## dump结果文件介绍

### MindSpore静态图场景

训练结束后，工具将dump的数据保存在dump_path参数指定的目录下。

- jit_level为O0/O1时

  dump结果目录请参见MindSpore官网中的《[同步Dump数据对象目录](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.0rc2/debug/dump.html#%E5%90%8C%E6%AD%A5dump%E6%95%B0%E6%8D%AE%E5%AF%B9%E8%B1%A1%E7%9B%AE%E5%BD%95)》。

- jit_level为O2时

  dump结果目录请参见MindSpore官网中的《[异步Dump数据对象目录](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.0rc2/debug/dump.html#%E5%BC%82%E6%AD%A5dump%E6%95%B0%E6%8D%AE%E5%AF%B9%E8%B1%A1%E7%9B%AE%E5%BD%95)》。

jit_level请参见[mindspore.set_context](https://www.mindspore.cn/docs/zh-CN/r2.3.0/api_python/mindspore/mindspore.JitConfig.html#mindspore-jitconfig)配置jit_config。

### MindSpore动态图场景

训练结束后，工具将dump的数据保存在dump_path参数指定的目录下。

dump结果目录结构示例如下：

```bash
├── dump_path
│   ├── step0
│   |   ├── rank0
│   |   │   ├── dump_tensor_data
|   |   |   |    ├── MintFunctional.relu.0.backward.input.0.npy
|   |   |   |    ├── Mint.abs.0.forward.input.0.npy
|   |   |   |    ├── Functional.split.0.forward.input.0.npy
|   |   |   |    ...
|   |   |   |    └── Tensor.__add__.0.forward.output.0.npy
│   |   |   ├── dump.json        # 保存前反向算子、算子的统计量信息或溢出算子信息。包含dump数据的API名称（命名格式为：`{api_type}_{api_name}_{API调用次数}_{前向反向}_{input/output}.{参数序号}`）、dtype、 shape、各数据的max、min、mean、L2norm统计信息以及当配置summary_mode="md5"时的md5数据。其中，“参数序号”表示该API下的第n个参数，例如1，则为第一个参数，若该参数为list格式，则根据list继续排序，例如1.1，表示该API的第1个参数的第1个子参数；L2norm表示L2范数（平方根）
│   |   |   ├── stack.json        # 算子调用栈信息
│   |   |   └── construct.json        # 分层分级结构，level为L1时，construct.json内容为空
│   |   ├── rank1
|   |   |   ├── dump_tensor_data
|   |   |   |   └── ...
│   |   |   ├── dump.json
│   |   |   ├── stack.json
|   |   |   └── construct.json
│   |   ├── ...
│   |   |
|   |   └── rank7
│   ├── step1
│   |   ├── ...
│   ├── step2
```

dump过程中，npy文件在对应算子或者模块被执行后就会落盘，而json文件则需要在正常执行PrecisionDebugger.stop()后才会写入完整数据，异常的程序终止会保存终止前被执行算子的相关npy文件，可能会导致json文件中数据丢失。

其中rank为设备上各卡的ID，每张卡上dump的数据会生成对应dump目录。非分布式场景下没有rank ID，目录名称为rank。

npy文件保存的前缀和MindSpore对应关系如下：

| 前缀           | MindSpore模块                |
| -------------- | ---------------------------- |
| Tensor         | mindspore.Tensor             |
| Functional     | mindspore.ops                |
| Mint           | mindspore.mint               |
| MintFunctional | mindspore.mint.nn.functional |

## 工具支持的API列表

msprobe工具维护固定的API支持列表，若需要删除或增加dump的API，可以在msprobe/mindspore/dump/hook_cell/support_wrap_ops.yaml文件内手动修改，如下示例：

```bash
ops:  # ops为算子类别，找到对应的类别，在该类别下按照下列格式删除或添加API
  - adaptive_avg_pool1d
  - adaptive_avg_pool2d
  - adaptive_avg_pool3d
```
