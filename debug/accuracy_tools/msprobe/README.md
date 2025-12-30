# 📖 msProbe

## 简介

**msProbe**是MindStudio Training Tools工具链下精度调试部分的工具包。主要包括精度预检、溢出检测和精度比对等功能，这些功能侧重不同的训练场景，可以定位模型训练中的精度问题。

## [版本说明](./docs/zh/release_notes.md)

包含msProbe的软件版本配套关系和软件包下载以及每个版本的特性变更说明。

## 环境部署

### 环境和依赖

- 硬件环境请参见《[昇腾产品形态说明](https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/quickstart/quickstart/quickstart_18_0002.html)》。
- 软件环境请参见《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/softwareinst/instg/instg_0000.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)》安装昇腾设备开发或运行环境，即toolkit软件包。

以上环境依赖请根据实际环境选择适配的版本。

### 工具安装

请参见[安装指导](./docs/zh/msprobe_install_guide.md)。

## 快速入门

为方便使用，本工具提供了统一、简易的程序接口：**PrecisionDebugger**。以 PyTorch框架为例，通过以下示例模板和**config.json**可以轻松使用各种功能。

```python
from msprobe.pytorch import PrecisionDebugger

debugger = PrecisionDebugger(config_path='./config.json')
...
debugger.start() # 一般在训练循环开头启动工具
... # 循环体
debugger.stop() # 一般在训练循环末尾结束工具。必须调用，否则可能导致精度数据落盘不全
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
**config.json** 的配置要求和各功能具体的使用指导请参见“[配置文件介绍](./docs/zh/dump/config_json_introduct.md)”。

详细快速入门可参见《训练场景工具快速入门》中的“[模型精度调试](https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/msquickstart/atlasquick_train_0004.html)”。

## 🚨 工具限制与注意事项

1. 工具读写的所有路径，如`config_path`、`dump_path`等，只允许包含大小写字母、数字、下划线、斜杠、点和短横线。

2. 出于安全性及权限最小化角度考虑，msprobe工具不应使用root等高权限账户使用，建议使用普通用户权限安装执行。

3. 使用msProbe工具前请确保执行用户的umask值大于等于0027，否则可能会导致工具生成的精度数据文件和目录权限过大。

4. 用户须自行保证使用最小权限原则，如给工具输入的文件要求other用户不可写，在一些对安全要求更严格的功能场景下还需确保输入的文件group用户不可写。

5. 使用工具前，建议先浏览[工具功能模块简介、适用场景和当前版本局限性](./docs/zh/limitations_and_precautions.md)，了解功能特性。

## 🧰 功能介绍

### PyTorch 场景

#### [训练前配置检查](./docs/zh/config_check_instruct.md)

训练前或精度比对前，对比两个环境下可能影响训练精度的配置差异。

#### [数据采集](./docs/zh/dump/pytorch_data_dump_instruct.md)

msProbe通过在训练脚本中添加PrecisionDebugger接口的方式对API执行精度数据dump操作。对应config.json中的 "statistics" 或 "tensor" task。

 config.json详细介绍请参见[介绍](./docs/zh/dump/config_json_introduct.md)和[示例](./docs/zh/dump/config_json_examples.md)。


#### [精度预检](./docs/zh/accuracy_checker/pytorch_accuracy_checker_instruct.md)

精度预检旨在昇腾NPU上扫描训练模型中的所有API进行API复现，给出精度情况的诊断和分析。对应config.json中的"run_ut" task。

#### [分级可视化构图比对](./docs/zh/accuracy_compare/pytorch_visualization_instruct.md)

该功能将msProbe工具dump的精度数据进行解析，还原模型图结构，实现模型各个层级的精度数据比对，方便用户理解模型结构、分析精度问题。

#### [精度比对](./docs/zh/accuracy_compare/pytorch_accuracy_compare_instruct.md)

该功能进行整网API级别的数据dump、精度比对，进而定位训练场景下的精度问题。

#### [数据解析](./docs/zh/other_functions/14.data_parse_PyTorch.md)

该功能用于比对前后两次NPU ACL层级dump数据的一致性。

#### [无标杆比对](./docs/zh/accuracy_compare/15.free_benchmarking_PyTorch.md)

该功能可以在没有标杆数据的情况下，检测模型训练中可能存在精度问题的API级别算子，并提供升精度和tocpu接口快速验证。

#### [梯度状态监测](./docs/zh/17.grad_probe.md)

该功能用于采集梯度数据并进行梯度相似度比对，可以精准定位出现问题所在的step。

#### [在线精度比对](./docs/zh/accuracy_compare/18.online_dispatch.md)

该功能是实现在训练过程中直接完成精度比对并输出比对结果的功能，是NPU与CPU之间的精度比对。

#### [训练状态监测](./docs/zh/monitor_instruct.md)

该功能收集和聚合模型训练过程中的网络层，优化器， 通信算子的中间值，帮助诊断模型训练过程中计算， 通信，优化器各部分出现的异常情况。

#### [单算子API自动生成脚本](./docs/zh/accuracy_compare/33.generate_operator_MindSpore.md)

该功能将msProbe工具dump的精度数据进行解析，自动生成单API脚本，用于复现整网中出现的算子问题，降低用户复现问题的成本，供开发分析算子问题。

#### [溢出检测与解析](./docs/zh/overflow_check/pytorch_overflow_check_instruct.md)

溢出检测用于采集溢出API或模块的精度数据，而溢出解析则是通过对溢出数据的分析，进一步判断是否为正常溢出。对应config.json中的"overflow_check" task。 
推荐直接使用[数据采集](#数据采集-1)功能采集统计量信息，检测溢出问题。

#### [checkpoint比对](./docs/zh/accuracy_compare/checkpoint_compare_instruct.md)

训练过程中或结束后，比较两个不同的checkpoint，评估模型相似度。

#### [强化学习数据采集](./docs/zh/other_functions/rl_collect_instruct.md)

灵活采集强化学习中重要关键过程数据，并支持比对。

#### [整网首个溢出节点分析](./docs/zh/other_functions/nan_analyze_instruct.md)

多rank场景下通过dump数据找到首个出现Nan或Inf的节点。

### MindSpore场景

#### [训练前配置检查](./docs/zh/config_check_instruct.md)

训练前或精度比对前，对比两个环境下可能影响训练精度的配置差异。

#### [数据采集](./docs/zh/dump/mindspore_data_dump_instruct.md)

msProbe通过在训练脚本中添加PrecisionDebugger接口的方式对API执行精度数据dump操作。对应config.json中的 "statistics" 或 "tensor" task。

 config.json详细介绍请参见[介绍](./docs/zh/dump/config_json_introduct.md)和[示例](./docs/zh/dump/config_json_examples.md)。


#### [精度预检](./docs/zh/accuracy_checker/mindspore_accuracy_checker_instruct.md)

精度预检旨在昇腾NPU上扫描训练模型中的所有API进行API复现，给出精度情况的诊断和分析。对应config.json中的"run_ut" task。

#### [分级可视化构图比对](./docs/zh/accuracy_compare/mindspore_visualization_instruct.md)

该功能将msProbe工具dump的精度数据进行解析，还原模型图结构，实现模型各个层级的精度数据比对，方便用户理解模型结构、分析精度问题。

#### [精度比对](./docs/zh/accuracy_compare/mindspore_accuracy_compare_instruct.md)

该功能进行整网API级别的数据dump、精度比对，进而定位训练场景下的精度问题。

#### [无标杆比对](./docs/zh/accuracy_compare/16.free_benchmarking_MindSpore.md)

该功能可以在没有标杆数据的情况下，检测模型训练中可能存在精度问题的API级别算子，并提供升精度和tocpu接口快速验证。

#### [梯度状态监测](./docs/zh/17.grad_probe.md)

该功能用于采集梯度数据并进行梯度相似度比对，可以精准定位出现问题所在的step。

#### [训练状态监测](./docs/zh/monitor_instruct.md)

该功能收集和聚合模型训练过程中的网络层，优化器， 通信算子的中间值，帮助诊断模型训练过程中计算， 通信，优化器各部分出现的异常情况。

#### [单算子API自动生成脚本](./docs/zh/accuracy_compare/33.generate_operator_MindSpore.md)

该功能将msProbe工具dump的精度数据进行解析，自动生成单API脚本，用于复现整网中出现的算子问题，降低用户复现问题的成本，供开发分析算子问题。

#### [数码关联](./docs/zh/other_functions/mindspore_code_mapping_instruct.md)

该功能只支持MindSpore静态图场景，用于将IR图与dump数据进行关联，获取dump数据和代码调用栈的关联关系。

#### [溢出检测与解析](./docs/zh/overflow_check/mindspore_overflow_check_instruct.md)

溢出检测用于采集溢出API或模块的精度数据，而溢出解析则是通过对溢出数据的分析，进一步判断是否为正常溢出。对应config.json中的"overflow_check" task。 
推荐直接使用[数据采集](#数据采集)功能采集统计量信息，检测溢出问题。

#### [checkpoint比对](./docs/zh/accuracy_compare/checkpoint_compare_instruct.md)

训练过程中或结束后，比较两个不同的checkpoint，评估模型相似度。

#### [强化学习数据采集](./docs/zh/other_functions/rl_collect_instruct.md)

灵活采集强化学习中重要关键过程数据，并支持比对。

### MSAdapter 场景

#### [数据采集](./docs/zh/dump/msadapter_data_dump_instruct.md)

 msProbe通过在训练脚本中添加PrecisionDebugger接口的方式对API执行精度数据dump操作。对应config.json中的 "statistics" 或 "tensor" task。

 config.json详细介绍请参见[介绍](./docs/zh/dump/config_json_introduct.md)和[示例](./docs/zh/dump/config_json_examples.md)。

#### [溢出检测与解析](./docs/zh/overflow_check/msadapter_overflow_check_instruct.md)

溢出检测用于采集溢出API或模块的精度数据，而溢出解析则是通过对溢出数据的分析，进一步判断是否为正常溢出。对应config.json中的"overflow_check" task。 
推荐直接使用[数据采集](#数据采集-2)功能采集统计量信息，检测溢出问题。

#### [checkpoint比对](./docs/zh/accuracy_compare/checkpoint_compare_instruct.md)

训练过程中或结束后，比较两个不同的checkpoint，评估模型相似度。

## 📑 补充材料

- [PyTorch场景的精度数据采集基线报告](./docs/zh/baseline/pytorch_data_dump_perf_baseline.md)

- [MindSpore场景的精度预检基线报告](./docs/zh/baseline/mindspore_accuracy_checker_perf_baseline.md)

- [MindSpore场景的精度数据采集基线报告](./docs/zh/baseline/mindspore_data_dump_perf_baseline.md)

- [训练状态监测工具标准性能基线报告](./docs/zh/baseline/monitor_perf_baseline.md)

- [无标杆工具场景验证和性能基线报告](./docs/zh/baseline/S02.report_free_benchmarking_validation_performance_baseline.md)

## ❓ FAQ

[FAQ for PyTorch](./docs/zh/faq.md)

## ❗ 免责声明

本工具建议执行用户与安装用户保持一致，如果您要使用root执行，请自行关注root高权限触及的安全风险。

## 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交issues，我们会尽快回复。感谢您的支持。

## 致谢

msProbe由华为公司的下列部门联合贡献：

- 昇腾计算MindStudio开发部
- 分布式并行计算实验室

感谢来自社区的每一个PR，欢迎贡献msProbe！

