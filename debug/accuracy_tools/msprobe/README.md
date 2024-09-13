# 📖 msprobe 使用手册

![version](https://img.shields.io/badge/version-1.0.3-blueviolet)
![python](https://img.shields.io/badge/python-3.8|3.9|3.10-blue)
![platform](https://img.shields.io/badge/platform-Linux-yellow)

[**msprobe**](./) 是 MindStudio Training Tools 工具链下精度调试部分的工具包。主要包括精度预检、溢出检测和精度比对等功能，目前适配 [PyTorch](https://pytorch.org/) 和 [MindSpore](https://www.mindspore.cn/) 框架。这些子工具侧重不同的训练场景，可以定位模型训练中的精度问题。

为方便使用，本工具提供了一个统一、简易的程序接口，**PrecisionDebugger**，以 PyTorch 框架为例，通过以下示例模板和 **config.json** 可轻松使用各种功能。

```python
from msprobe.pytorch import PrecisionDebugger  

debugger = PrecisionDebugger(config_path='./config.json')
...
debugger.start() # 一般在训练循环开头启动工具
... # 循环体
debugger.stop() # 一般在训练循环末尾结束工具
debugger.step() # 在训练循环的最后需要重置工具，非循环场景不需要
```

除了在训练脚本中调用接口函数，还可以通过命令行使用 **msprobe** 的其他功能，具体的使用规则和 **config.json** 的配置要求详见以下章节。

## ⚙️ [安装](./docs/01.installation.md)

## 🛠️ config.json [介绍](./docs/02.config_introduction.md) 和 [示例](./docs/03.config_examples.md)

## 🧰 主要功能

### 1 数据采集

msprobe 通过在训练脚本中添加 PrecisionDebugger 接口的方式对 API 执行精度数据 dump 操作，对应 config.json 中的 task 为 statistics 或 tensor。

[PyTorch 场景的数据采集](./docs/05.data_dump_PyTorch.md)

[MindSpore 场景的数据采集](./docs/06.data_dump_MindSpore.md)

### 2 精度预检

精度预检旨在昇腾 NPU 上扫描训练模型中的所有 API 进行 API 复现，给出精度情况的诊断和分析。对应 config.json 中的 task 为 run_ut。

PyTorch 场景的[离线预检](./docs/07.accuracy_checker_PyTorch.md)和[在线预检](./docs/08.accuracy_checker_online_PyTorch.md)

MindSpore 动态图场景的[离线预检](./docs/09.accuracy_checker_MindSpore.md)

### 3 精度比对

该功能进行 PyTorch 整网 API 粒度的数据 dump、精度比对，进而定位训练场景下的精度问题。

[PyTorch 场景的精度比对](./docs/10.accuracy_compare_PyTorch.md)

[MindSpore 场景的精度比对](./docs/11.accuracy_compare_MindSpore.md)

### 4 溢出检测与解析

溢出检测与解析是在执行精度数据 dump 时，判断是否存在输入正常但输出存在溢出的 API，从而判断是否为正常溢出。对应 config.json 中的 overflow_check。

[PyTorch 场景的溢出检测与解析](./docs/12.overflow_check_PyTorch.md)

[MindSpore 场景的溢出检测与解析](./docs/13.overflow_check_MindSpore.md)

### 5 数据解析

该功能用于比对前后两次 NPU ACL 层级 dump 数据的一致性。

[PyTorch 场景的数据解析](./docs/14.data_parse_PyTorch.md)

### 6 无标杆比对

[PyTorch 场景的无标杆比对](./docs/15.free_benchmarking_PyTorch.md)（待补充）

[MindSpore 场景的无标杆比对](./docs/16.overflow_check_MindSpore.md)（待补充）

### 7 梯度状态监测

本功能用于采集梯度数据并进行梯度相似度比对，可以精准定位出现问题的 step。

[兼容 PyTorch 和 MindSpore 框架的梯度监测](./docs/17.grad_probe.md)

## 🌟 新版本特性

【精度预检】
- 落盘数据小。
- 支持随机生成模式和真实数据模式。
- 单 API 测试，排除整网中的累计误差问题。

【梯度检测】
- 使用便捷，无需在训练流程里插入代码。
- 可以精准定位问题出现的 step。

## 📑 补充材料

[msprobe 标准性能基线报告](./docs/S01.report_msprobe_dump_standard_performance_baseline.md)

[无标杆工具场景验证和性能基线报告](./docs/S02.report_free_benchmarking_validation_performance_baseline.md)

## ❓ FAQ

[FAQ for PyTorch](./docs/FAQ_PyTorch.md)

FAQ for MindSpore
