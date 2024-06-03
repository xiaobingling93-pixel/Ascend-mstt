# dump配置文件说明

dump配置文件主要为PrecisionDebugger接口执行dump操作时，调用的可选配置，故下列参数均为可选参数，且均有默认值，当PrecisionDebugger接口未指定该配置文件时，dump操作使用该文件的默认配置。配置文件详见[config.json](./config.json)。

## 参数说明

**通用配置参数**

| 参数名           | 说明                                                         | 是否必选 |
| ---------------- | ------------------------------------------------------------ | -------- |
| task             | dump的任务类型。可取值"statistics"（仅dump API统计信息，默认值）、"tensor"（dump API统计信息和完全复刻整网的API运行情况的真实数据）、"overflow_check"（溢出检测）。配置示例："task": "tensor"。 | 否       |
| dump_path        | 设置dump数据目录路径。配置示例："dump_path": "./dump_path"。 | 是       |
| rank             | 指定对某张卡上的数据进行dump，默认未配置（表示dump所有卡的数据），须根据实际卡的Rank ID配置。应配置为大于0的正整数，且须根据实际卡的Rank ID配置，若所配置的值大于实际训练所运行的卡的Rank ID，则dump数据为空，比如当前环境Rank ID为0到7，实际训练运行0到3卡，此时若配置Rank ID为4或不存在的10等其他值，此时dump数据为空。配置示例："rank": [1]。 | 否       |
| step             | 指定dump某个step的数据，默认未配置，表示dump所有step数据。dump特定step时，须指定为训练脚本中存在的step。step为list格式，可配置逐个step，例如："step": [0,1,2]。 | 否       |
| level            | dump级别，根据不同级别dump不同数据。可取值"L0"（dump module模块级精度数据，仅PyTorch场景支持，使用背景详见“**模块级精度数据dump说明**”）、"L1"（dump API级精度数据，默认值）、"L2"（dump kernel级精度数据，仅MindSpore场景支持）、"mix"（dump module模块级和API级精度数据）。配置示例："level": "L1"。 | 否       |
| seed             | 随机种子数，默认值为：1234。通过固定随机数保证模型的输入或输出一致，可固定的随机数详见“**固定随机数范围**”。配置示例："seed": 1234。 | 否       |
| is_deterministic | 确定性计算模式。可取值true（开启）或false（关闭），默认关闭。配置示例："is_deterministic": true。<br/>即使在相同的硬件和输入下，API多次执行的结果也可能不同，开启确定性计算是为了保证在相同的硬件和输入下，API多次执行的结果相同。<br/>确定性计算会导致API执行性能降低，建议在发现模型多次执行结果不同的情况下开启。<br/>rnn类算子、ReduceSum、ReduceMean等算子可能与确定性计算存在冲突，若开启确定性计算后多次执行的结果不相同，则考虑存在这些算子。 | 否       |

根据task参数取值的不同，可以配置如下参数：

**task配置为statistics**

| 参数名       | 说明                                                         | 是否必选 |
| ------------ | ------------------------------------------------------------ | -------- |
| scope        | PyTorch场景dump范围，默认未配置。需要在[]内配置两个模块名或API名，用于锁定区间，dump该范围内的数据。配置示例："scope": ["MyModuleOP1", "MyModuleOP2"]。与level参数取值相关，level为L0和mix级别时，可配置模块名；level为L1级别时，可配置API名。 | 否       |
| list         | 自定义dump范围，默认未配置。包含如下配置方法：<br>        PyTorch场景配置具体的API全称，dump该API数据。配置示例："list": ["Tensor.permute.1.forward", "Tensor.transpose.2.forward", "Torch.relu.3.backward"]。<br/>        PyTorch场景指定某一类API，dump某一类的API级别输入输出数据。配置示例："list": ["relu"]。<br/>        MindSpore场景配置kernel_name，指定算子名dump。 | 否       |
| data_mode    | dump数据过滤。可取值"all"、"forward"、"backward"、"input"和"output"，表示仅保存dump的数据中文件名包含"forward"、"backward"、"input"和"output"的前向、反向、输入或输出的dump文件。配置示例"data_mode": ["backward"]或"data_mode": ["forward", "backward"]。默认为["all"]，即保存所有dump的数据。除了all参数只能单独配置外，其他参数可以自由组合。 | 否       |
| summary_mode | 控制dump文件输出的模式，可取值md5（dump仅输出包含md5值的dump.json文件，用于验证数据的完整性）、statistics（dump仅输出包含API统计信息的dump.json文件，默认值）。配置示例："summary_mode": "md5"。 | 否       |

**task配置为tensor**

| 参数名         | 说明                                                         | 是否必选 |
| -------------- | ------------------------------------------------------------ | -------- |
| scope          | PyTorch场景dump范围，默认未配置。需要在[]内配置两个模块名或API名，用于锁定区间，dump该范围内的数据。配置示例："scope": ["MyModuleOP1", "MyModuleOP2"]。与level参数取值相关，level为L0和mix级别时，可配置模块名；level为L1级别时，可配置API名。 | 否       |
| list           | 自定义dump范围，默认未配置。包含如下配置方法：<br>        PyTorch场景配置具体的API全称，dump该API数据。配置示例："list": ["Tensor.permute.1.forward", "Tensor.transpose.2.forward", "Torch.relu.3.backward"]。<br/>        PyTorch场景指定某一类API，dump某一类的API级别输入输出数据。配置示例："list": ["relu"]。<br/>        PyTorch场景配置kernel_api，dump前向和反向API的kernel_api级别数据，其中dump反向API时需要配置**backward_input**参数。前向API配置示例："list": ["Tensor.permute.1.forward"]；反API配置示例："list": ["Tensor.permute.1.forward"], "backward.input": "./npu_dump/step0/rank0/Functional.conv2d.1.backward.input.0.pt"]。<br/>        配置kernel_name，MindSpore为指定算子名dump，PyTorch需要指定API名称以及level为L2（PyTorch场景暂不支持）。 | 否       |
| backward_input | 该输入文件为首次运行训练dump得到反向API输入的dump文件，默认未配置。例如若需要dump Functional.conv2d.1 API的反向过程的输入输出，则需要在dump目录下查找命名包含Functional.conv2d.1、backward和input字段的dump文件。配置示例："backward_input": "./npu_dump/step0/rank0/Functional.conv2d.1.backward.input.0.pt"] | 否       |
| data_mode      | dump数据过滤。可取值"all"、"forward"、"backward"、"input"和"output"，表示仅保存dump的数据中文件名包含"forward"、"backward"、"input"和"output"的前向、反向、输入或输出的dump文件。配置示例"data_mode": ["backward"]或"data_mode": ["forward", "backward"]。默认为["all"]，即保存所有dump的数据。除了all参数只能单独配置外，其他参数可以自由组合。 | 否       |
| file_format    | MindSpore场景真实tensor数据的保存格式，可取值"bin"（dump的tensor文件为二进制格式）、"npy"（dump的tensor文件后缀为.npy，默认值）。 | 否       |

**task配置为overflow_check**

| 参数名        | 说明                                                         | 是否必选 |
| ------------- | ------------------------------------------------------------ | -------- |
| overflow_nums | 暂未支持。控制溢出次数，表示第N次溢出时，停止训练，过程中检测到溢出API对应kernel数据均dump。配置示例："overflow_nums": 3。默认为1，即检测到1次溢出，训练停止，配置为-1时，表示持续检测溢出直到训练结束。 | 否       |
| check_mode    | MindSpore场景kernel级别的溢出检测，可取值"aicore"（开启AI Core的溢出检测）、"atomic"（开启Atomic的溢出检测）、"all"（开启AI Core和Atomic的溢出检测，默认值）。配置示例"check_mode": "aicore"。 | 否       |

## 模块级精度数据dump说明

大模型场景下，通常不是简单的利用自动迁移能力实现GPU到NPU的训练脚本迁移，而是会对NPU网络进行一系列针对性的适配，因此，常常会造成迁移后的NPU模型存在部分子结构不能与GPU原始模型完全对应。模型结构不一致导致API调用类型及数量不一致，若直接按照API粒度进行精度数据dump和比对，则无法完全比对所有的API。

本节介绍的功能是对模型中的大粒度模块进行数据dump，使其比对时，对于无法以API粒度比对的模块可以直接以模块粒度进行比对。

模块指的是继承自nn.Module类模块，通常情况下这类模块就是一个小模型，可以被视为一个整体，dump数据时以模块为粒度进行dump。

## 固定随机数范围

seed_all函数可固定随机数的范围如下表。

| API                                      | 固定随机数                  |
| ---------------------------------------- | --------------------------- |
| os.environ['PYTHONHASHSEED'] = str(seed) | 禁止Python中的hash随机化    |
| random.seed(seed)                        | 设置random随机生成器的种子  |
| np.random.seed(seed)                     | 设置numpy中随机生成器的种子 |
| torch.manual_seed(seed)                  | 设置当前CPU的随机种子       |
| torch.cuda.manual_seed(seed)             | 设置当前GPU的随机种子       |
| torch.cuda.manual_seed_all(seed)         | 设置所有GPU的随机种子       |
| torch_npu.npu.manual_seed(seed)          | 设置当前NPU的随机种子       |
| torch_npu.npu.manual_seed_all(seed)      | 设置所有NPU的随机种子       |
| torch.backends.cudnn.enable=False        | 关闭cuDNN                   |
| torch.backends.cudnn.benchmark=False     | cuDNN确定性地选择算法       |
| torch.backends.cudnn.deterministic=True  | cuDNN仅使用确定性的卷积算法 |

需要保证CPU或GPU以及NPU的模型输入完全一致，dump数据的比对才有意义，seed_all并不能保证模型输入完全一致，如下表所示场景需要保证输入的一致性。

| 场景            | 固定方法      |
| --------------- | ------------- |
| 数据集的shuffle | 关闭shuffle。 |
| dropout         | 关闭dropout。 |

关闭shuffle示例：

```Python
train_loader = torch.utils.data.DataLoader(
	train_dataset,
	batch_size = batch_size,
	shuffle = False,
	num_workers = num_workers
)
```

关闭dropout：

在使用from ptdbg import *后，工具会自动将torch.nn.functional.dropout、torch.nn.functional.dropout2d、torch.nn.functional.dropout3d、torch.nn.Dropout、torch.nn.Dropout2d、torch.nn.Dropout3d的接口参数p置为0。

## 配置示例

以下示例包含当前支持的所有场景可配置的完整参数，参数取值均为默认值。

### PyTorch场景task配置为statistics

```json
{
    "task": "statistics",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",
    "seed": 1234,
    "is_deterministic": false,

    "statistics": {
        "scope": [], 
        "list": [],
        "data_mode": ["all"],
        "summary_mode": "statistics"
    }
}
```

### PyTorch场景task配置为tensor

```json
{
    "task": "tensor",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",
    "seed": 1234,
    "is_deterministic": false,

    "tensor": {
        "scope": [],
        "list":[],
        "data_mode": ["all"],
        "backward_input": ""
    }
}
```

### PyTorch场景task配置为overflow_check

```json
{
    "task": "overflow_check",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",
    "seed": 1234,
    "is_deterministic": false,

    "overflow_check": {
        "overflow_nums": 1
    }
}
```

### MindSpore场景task配置为statistics

```json
{
    "task": "statistics",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",
    "seed": 1234,
    "is_deterministic": false,

    "statistics": {
        "list": [],
        "data_mode": ["all"],
        "summary_mode": "statistics"
    }
}
```

### MindSpore场景task配置为tensor

```json
{
    "task": "tensor",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",
    "seed": 1234,
    "is_deterministic": false,

    "tensor": {
        "list":[],
        "data_mode": ["all"],
        "backward_input": ""
    }
}
```

### MindSpore场景task配置为overflow_check

```json
{
    "task": "overflow_check",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",
    "seed": 1234,
    "is_deterministic": false,

    "overflow_check": {
        "overflow_nums": 1,
        "check_mode": "all"
    }
}
```

