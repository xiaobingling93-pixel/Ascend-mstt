# **精度数据采集**

atat工具主要通过在训练脚本内添加dump接口并启动训练的方式来采集精度数据。

执行dump操作需要安装atat工具。详见《[MindStudio精度调试工具](../../README.md)》的“工具安装”章节。

## dump接口介绍

### PrecisionDebugger

**功能说明**

通过加载dump配置文件的方式来确定dump操作的详细配置。

可以在from atat.pytorch import PrecisionDebugger和模型初始化之间的任意位置添加该接口。

**原型**

```Python
PrecisionDebugger(config_path=None, task=None, dump_path=None, level=None, model=None, step=None)
```

说明：上述参数除config_path和model外，其他参数均在[config.json](../../config)文件中可配，此处的参数优先级高于config.json文件中的配置，而config.json文件可以配置更多参数，若需要进行更多场景的精度数据dump，建议配置[config.json](../../config)文件。

**参数说明**

| 参数名      | 说明                                                         | 是否必选 |
| ----------- | ------------------------------------------------------------ | -------- |
| config_path | 指定dump配置文件路径，String类型。参数示例："./config.json"。未配置该路径时，默认使用../../config目录下的config.json文件的默认配置。 | 否       |
| task        | dump的任务类型，String类型。可取值"statistics"（仅dump API统计信息）、"tensor"（dump API统计信息和完全复刻整网的API运行情况的真实数据）、"overflow_check"（溢出检测），默认未配置，取"statistics"，参数示例：task="tensor"。 | 否       |
| dump_path   | 设置dump数据目录路径，String类型。参数示例：dump_path="./dump_path"。 | 是       |
| level       | dump级别，根据不同级别dump不同数据，String类型。可取值：<br>        "L0"：dump module模块级精度数据，仅PyTorch场景支持”。<br/>        "L1"：dump API级精度数据，默认值。<br/>        "L2"：dump kernel级精度数据。<br/>        "mix"：dump module模块级和API级精度数据。<br/>配置示例：level="L1"。 | 否       |
| model       | 指定具体的torch.nn.Module，默认未配置，level配置为"L0"或"mix"时必须配置该参数。配置示例参见“**model配置代码示例**”。 | 否       |
| step        | 指定dump某个step的数据，list[int]类型。默认未配置，表示dump所有step数据。dump特定step时，须指定为训练脚本中存在的step。step为list格式，可配置逐个step，例如：step=[0,1,2]。 | 否       |

#### model配置代码示例

示例中定义了一个nn.Module类型的简单网络，在进行数据dump时使用原型函数PrecisionDebugger并传入config_path参数和model参数，其中model参数传入数据的类型为torch.nn.Module类型或torch.nn.Module子类型。

```python
#根据需要import包
import os
import torch
import torch.nn as nn
import torch_npu
import torch.nn.functional as F
from atat.pytorch import PrecisionDebugger

torch.npu.set_device("npu:0")
#定义一个简单的网络
class ModuleOP(nn.Module)
def __init__(self) -> None:
    super().__init__()
    self.linear_1 = nn.Linear(in_features=8,out_features=4)
    self.linear_2 = nn.Linear(in_features=4,out_features=2)
def forward(self,x):
    x1 = self.linear_1(x)
    x2 = self.linear_2(x1)
    r1 = F.relu(x2)
    return r1

if __name__ == "__main__"
module = ModuleOP()

#注册工具
debugger = PrecisionDebugger('./config.json',model=module)
debugger.start()
x = torch.randn(10,8)
out = module(x)
loss = out.sum()
loss.backward()
debugger.stop()
```

### start函数

**功能说明**

启动函数。

在模型初始化之后的任意位置添加。

**原型**

```Python
debugger.start()
```

该函数为类函数，可以使用debugger.start()也可以使用PrecisionDebugger.start()。

### stop函数

**功能说明**

停止函数。

在**start**函数之后的任意位置添加。

**原型**

```Python
debugger.stop()
```

该函数为类函数，可以使用debugger.stop()也可以使用PrecisionDebugger.stop()。

### step函数

**功能说明**

结束标识。

在最后一个**stop**函数后或一个step结束的位置添加。

**原型**

```Python
debugger.step()
```

该函数为类函数，可以使用debugger.step()也可以使用PrecisionDebugger.step()。

## 示例代码

```Python
from atat.pytorch import PrecisionDebugger
debugger = PrecisionDebugger(config_path="./config.json", dump_path="./dump_path")
# 请勿将以上初始化流程插入到循环代码中

# 模型初始化
# 下面代码也可以用PrecisionDebugger.start()和PrecisionDebugger.stop()
debugger.start()

# 需要dump的代码片段1

debugger.stop()
debugger.start()

# 需要dump的代码片段2

debugger.stop()
debugger.step()
```

## dump结果文件介绍

训练结束后，工具将dump的数据保存在dump_path参数指定的目录下。

dump结果目录结构示例如下：

```Python
├── dump_path
│   ├── step0
│   |   ├── rank0
│   |   │   ├── dump_tensor_data
|   |   |   |    ├── Tensor.permute.1.forward.pt
|   |   |   |    ├── MyModule.0.forward.input.pt        # 开启模块级精度数据dump时存在模块级的dump数据文件
|   |   |   |    ...
|   |   |   |    └── Fcuntion.linear.5.backward.output.pt
│   |   |   ├── dump.json        # 保存前反向算子、算子的统计量信息或溢出算子信息。包含dump数据的API名称（命名格式为：`{api_type}_{api_name}_{API调用次数}_{前向反向}_{input/output}.{参数序号}`）、dtype、 shape、各数据的max、min、mean、L2norm统计信息以及当配置summary_mode="md5"时的md5数据。其中，“参数序号”表示该API下的第n个参数，例如1，则为第一个参数，若该参数为list格式，则根据list继续排序，例如1.1，表示该API的第1个参数的第1个子参数；L2norm表示2范数（平方根）
│   |   |   ├── stack.json        # 算子调用栈信息
│   |   |   └── construct.json        # 分层分级结构
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

dump过程中，pt文件在对应算子或者模块被执行后就会落盘，而json文件则需要在正常执行PrecisionDebugger.stop()或set_dump_switch("OFF")后才会被落盘保存，异常的程序终止会保存终止前被执行算子的相关pt文件，但是不会生成json文件。

其中`dump_{version}`为默认命名，debugger方式dump不支持修改该文件夹名称；rank为设备上各卡的ID，每张卡上dump的数据会生成对应dump目录。

pt文件保存的前缀和PyTorch对应关系如下：

| 前缀        | Torch模块           |
| ----------- | ------------------- |
| Tensor      | torch.Tensor        |
| Torch       | torch               |
| Functional  | torch.nn.functional |
| NPU         | NPU亲和算子         |
| VF          | torch._VF           |
| Aten        | torch.ops.aten      |
| Distributed | torch.distributed   |

## 工具支持的API列表

atat工具维护固定的API支持列表，若需要删除或增加dump的API，可以在atat/pytorch/hook_module/support_wrap_ops.yaml文件内手动修改，如下示例：

```Python
functional:  # functional为算子类别，找到对应的类别，在该类别下按照下列格式删除或添加API
  - conv1d
  - conv2d
  - conv3d
```

# FAQ

[FAQ](./FAQ.md)
