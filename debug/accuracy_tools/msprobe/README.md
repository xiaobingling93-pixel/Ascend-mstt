# 📖 msprobe 使用手册

![version](https://img.shields.io/badge/version-1.0.4-blueviolet)
![python](https://img.shields.io/badge/python-3.8|3.9|3.10-blue)
![platform](https://img.shields.io/badge/platform-Linux-yellow)

**msprobe** 是 MindStudio Training Tools 工具链下精度调试部分的工具包。主要包括精度预检、溢出检测和精度比对等功能，目前适配 [PyTorch](https://pytorch.org/) 和 [MindSpore](https://www.mindspore.cn/) 框架。这些子工具侧重不同的训练场景，可以定位模型训练中的精度问题。

为方便使用，本工具提供了统一、简易的程序接口：**PrecisionDebugger**。以 PyTorch 框架为例，通过以下示例模板和 **config.json** 可以轻松使用各种功能。

```python
from msprobe.pytorch import PrecisionDebugger  

debugger = PrecisionDebugger(config_path='./config.json')
...
debugger.start() # 一般在训练循环开头启动工具
... # 循环体
debugger.stop() # 一般在训练循环末尾结束工具
debugger.step() # 在训练循环的最后需要重置工具，非循环场景不需要
```

此外，根据以下规则，可以通过环境变量设置日志级别。
- MSPROBE_LOG_LEVEL=4，不打印任何日志；
- MSPROBE_LOG_LEVEL=3，仅打印 ERROR；
- MSPROBE_LOG_LEVEL=2，仅打印 WARNING、ERROR；
- MSPROBE_LOG_LEVEL=1，仅打印 INFO、WARNING、ERROR（默认配置）；
- MSPROBE_LOG_LEVEL=0，打印 DEBUG、INFO、WARNING、ERROR。

例如在 shell 脚本：

```shell
export MSPROBE_LOG_LEVEL={x}
```
**config.json** 的配置要求和各功能具体的使用指导详见后续章节。

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

[PyTorch 场景的无标杆比对](./docs/15.free_benchmarking_PyTorch.md)

[MindSpore 场景的无标杆比对](./docs/16.free_benchmarking_MindSpore.md)

### 7 梯度状态监测

本功能用于采集梯度数据并进行梯度相似度比对，可以精准定位出现问题的 step。

[兼容 PyTorch 和 MindSpore 框架的梯度监测](./docs/17.grad_probe.md)

### 8 在线精度比对

在线精度比对是实现在PyTorch训练过程中直接完成精度比对并输出比对结果的功能，是NPU与CPU之间的精度比对。

[PyTorch 场景的在线精度比对](./docs/18.online_dispatch.md)

## 🌟 新版本特性

若查看历史版本特性，请点击[安装](./docs/01.installation.md)。

【数据采集】
- 支持 config.json 中的 step 传入范围；
- 优化了指定 step 的机制，指定 step 结束后工具不再采集数据，但训练会继续运行。工具结束运行后，日志提示信息如下：
    ```bash
    ****************************************
    *      msprobe ends successfully.      *
    ****************************************
    ```
    注：在多卡场景，每张卡进程训练到指定 step 之后都会打印一次上述信息。

【精度预检】
- 在 PyTorch 场景，支持部分 NPU 融合算子预检。

【精度比对】
- 解决了使用 MindSpore 需要安装 PyTorch 的问题。

【无标杆比对】
- 补充在 PyTorch 场景的性能基线报告；
- 支持 MindSpore 场景的 change_value 扰动模式。

## 📑 补充材料

[无标杆比对功能在 PyTorch 场景的性能基线报告](./docs/S02.report_free_benchmarking_validation_performance_baseline.md)

## ❗ 免责声明
本工具建议执行用户与安装用户保持一致，如果您要使用 root 执行，请自行关注 root 高权限触及的安全风险。

## ❓ FAQ

[FAQ for PyTorch](./docs/FAQ_PyTorch.md)
