# MindStudio Training Tools

## 最新消息

[2026.2.25]：Tinker并行策略自动寻优系统正式开源，具体请参见[Tinker](https://gitcode.com/Ascend/mstt/tree/master/profiler/tinker)。

[2026.1.12]：[mstt 仓库License变更通知](https://gitcode.com/Ascend/mstt/discussions/1)

[2025.12.31]：MindStudio昇腾平台训练工具链全面开源，涉及如下代码仓。

- [MindStudio-Profiler](https://gitcode.com/Ascend/msprof)

  构建昇腾全场景性能调优基础能力，支持采集CANN和NPU性能数据，提升昇腾设备性能调优效率。

- [MindStudio-Profiler-Analyze](https://gitcode.com/Ascend/msprof-analyze)

  昇腾性能分析工具，基于采集的性能数据进行分析，提供昇腾设备性能瓶颈快速识别能力。

- [MindStudio-MemScope](https://gitcode.com/Ascend/msmemscope)

  针对昇腾显存调试调优场景的专用工具，提供整网级多维度显存数据采集、自动诊断、优化分析能力。

- [MindStudio-Probe](https://gitcode.com/Ascend/msprobe)

  模型开发精度调试环节使用的工具包，是针对昇腾提供的全场景精度工具链，帮助用户提高模型精度定位效率。

  如果安装的是8.x版本及之前的MindStudio-Probe，请参考[MindStudio-Probe-8.x](debug/accuracy_tools/msprobe/README.md)

- [MindStudio-Monitor](https://gitcode.com/Ascend/msmonitor)

  一站式在线监控工具，支持落盘和在线性能数据采集，提供集群场景性能监测及定位能力。

- [MindStudio-Profilier-Tools-Interface](https://gitcode.com/Ascend/mspti)

  MindStudio针对Ascend设备提出的一套Profiling API，用户可以通过msPTI构建针对NPU应用程序的工具，用于分析应用程序的性能。
  
- [MindStudio-Insight](https://gitcode.com/Ascend/msinsight)

  MindStudio Insight可视化工具，支持系统级、算子级、服务化等多场景多维度性能分析，深度剖析性能数据，帮助开发者完成性能诊断。

## 简介

MindStudio Training Tools（MindStudio训练工具链，msTT）聚焦您在模型迁移、模型开发中遇到的痛点问题，提供全流程的工具链，通过提供分析迁移工具、精度调试工具、性能调优工具三大主力工具包，帮助您解决开发过程中迁移困难、Loss跑飞、性能不达标或劣化等问题，让您轻松解决精度和性能问题，开启乐趣十足的极简开发之旅。

**模型训练开发全流程**

![training_process](docs/zh/figures/training_process.png)

## 目录结构

关键目录如下。

```ColdFusion
├── docs              // 文档目录
├── msfmktransplt     // MindStudio分析迁移工具源码目录
├── scripts           // 存放安装卸载升级脚本
├── msinsight         // MindStudio可视化调优工具源码目录
├── msmemscope        // MindStudio内存检测工具源码目录
├── msmoniter         // MindStudio一站式在线监控工具源码目录
├── msprobe           // MindStudio精度调试工具源码目录
├── msprof            // MindStudio模型调优工具源码目录
├── msprof-analyze    // MindStudio性能分析工具源码目录
├── mspti             // MindStudio Profiling Tools Interface工具源码目录
└── README.md         // 整体仓代码说明
```

## 版本说明

msTT的版本说明包含msTT的软件版本配套关系以及每个版本的特性变更说明，具体参见[版本说明](docs/zh/release_notes.md)。

## 快速入门

msTT工具快速入门当前提供在PyTorch和MindSpore训练场景中，通过一个可执行样例，串联使用分析迁移、精度调试和性能调优流程对应的工具，帮助用户快速上手。

具体参见《[PyTorch场景msTT工具快速入门](docs/zh/pytorch_mstt_quick_start.md)》和《[MindSpore场景msTT工具快速入门](docs/zh/mindspore_mstt_quick_start.md)》。

## 功能介绍

### 分析迁移工具

[MindStudio Analysis and Migration Tool（MindStudio分析迁移工具，msfmktransplt）](./msfmktransplt/docs/zh/msfmktransplt_instruct.md)

PyTorch训练脚本一键式迁移至昇腾NPU的功能，开发者可做到少量代码修改或零代码完成迁移。

### 精度调试工具

- [MindStudio Probe（MindStudio精度调试工具，msProbe）](https://gitcode.com/Ascend/msprobe)

  模型开发精度调试环节使用的工具包，是针对昇腾提供的全场景精度工具链，帮助用户提高模型精度定位效率。

  如果安装的是8.x版本及之前的MindStudio-Probe，请参考[MindStudio-Probe-8.x](debug/accuracy_tools/msprobe/README.md)

- [Tensorboard](https://gitcode.com/Ascend/msprobe/tree/master/plugins/tb_graph_ascend)

  Tensorboard支持模型结构进行分级可视化展示的插件tb-graph-ascend

  可将模型的层级关系、精度数据进行可视化，并支持将调试模型和标杆模型进行分视图展示和关联比对，方便用户快速定位精度问题。

  **注：MindStudio昇腾平台训练工具链现已全面开源，模型分级可视化插件已经并入MindStudio Probe仓库，此仓库相关内容后续不再维护演进，建议使用最新版本，请参考**[tb_graph_ascend](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/accuracy_compare/pytorch_visualization_instruct.md)；如果安装的是8.x版本及之前的MindStudio-Probe，请参考[tb_graph_ascend-8.x](plugins/tensorboard-plugins/tb_graph_ascend)。

### 性能调优工具

- [MindStudio Profiler（MindStudio模型调优工具，msProf）](https://gitcode.com/Ascend/msprof)

  构建昇腾全场景性能调优基础能力，支持采集CANN和NPU性能数据，提升昇腾设备性能调优效率。

- [MindStudio Profiler Analyze（MindStudio性能分析工具，msprof-analyze）](https://gitcode.com/Ascend/msprof-analyze)

  昇腾性能分析工具，基于采集的性能数据进行分析，提供昇腾设备性能瓶颈快速识别能力。

- [msMemScope（MindStudio内存检测工具）](https://gitcode.com/Ascend/msmemscope)

  针对昇腾显存调试调优场景的专用工具，提供整网级多维度显存数据采集、自动诊断、优化分析能力。

- [MindStudio Monitor（MindStudio一站式在线监控工具，msMonitor）](https://gitcode.com/Ascend/msmonitor)

  一站式在线监控工具，支持落盘和在线性能数据采集，提供集群场景性能监测及定位能力。
  
- [MindStudio Profiling Tools Interface（msPTI）](https://gitcode.com/Ascend/mspti)

  MindStudio针对Ascend设备提出的一套Profiling API，用户可以通过msPTI构建针对NPU应用程序的工具，用于分析应用程序的性能。
  
- [MindStudio Insight（MindStudio可视化调优工具，msInsight）](https://gitcode.com/Ascend/msinsight)

  MindStudio Insight可视化工具，支持系统级、算子级、服务化等多场景多维度性能分析，深度剖析性能数据，帮助开发者完成性能诊断。
  
- [bind_core](https://gitcode.com/Ascend/mstt/tree/master/profiler/affinity_cpu_bind)

  绑核脚本，支持非侵入修改工程代码，实现一键式绑核功能。
- [Tinker](https://gitcode.com/Ascend/mstt/tree/master/profiler/tinker)

  Tinker大模型并行策略自动寻优系统，根据提供的训练脚本，进行单节点NPU性能测量，推荐高性能并行策略训练脚本。

## 安全声明

描述msTT产品的安全加固信息、公网地址信息等内容，具体请参见[安全声明](docs/zh/security_statement.md)。

## 分支维护策略

1. MindStudio Training Tools工具版本分支的维护阶段如下：

   | **状态**            | **时间** | **说明**                                         |
   | ------------------- | -------- | ------------------------------------------------ |
   | 计划                | 1—3个月  | 计划特性                                         |
   | 开发                | 3个月    | 开发特性                                         |
   | 维护                | 6—12个月 | 合入所有已解决的问题并发布版本                   |
   | 无维护              | 0—3个月  | 合入所有已解决的问题，无专职维护人员，无版本发布 |
   | 生命周期终止（EOL） | N/A      | 分支不再接受任何修改                             |

2. MindStudio Training Tools分支版本号命名规则如下：

   msTT仓每年发布4个版本，每个版本都将对应一个分支；以v6.0为例，其将对应v6.0.RC1、v6.0.RC2、v6.0.RC3以及v6.0.0四个版本，在仓库中将存在与之对应的分支。

   | **分支**      | **状态** | **发布日期** | **后续状态**               | **EOL日期** |
   | ------------- | -------- | ------------ | ------------------------ | ----------- |
   | **v6.0.0** | 维护     | 2023.12.12   | 预计2024.12.12起无维护    |             |

## 免责声明

- 本工具仅供调试和开发之用，使用者需自行承担使用风险，并理解以下内容：
  - 数据处理及删除：用户在使用本工具过程中产生的数据属于用户责任范畴。建议用户在使用完毕后及时删除相关数据，以防信息泄露。
  - 数据保密与传播：使用者了解并同意不得将通过本工具产生的数据随意外发或传播。对于由此产生的信息泄露、数据泄露或其他不良后果，本工具及其开发者概不负责。
  - 用户输入安全性：用户需自行保证输入的命令行的安全性，并承担因输入不当而导致的任何安全风险或损失。对于由于输入命令行不当所导致的问题，本工具及其开发者概不负责。
- 免责声明范围：本免责声明适用于所有使用本工具的个人或实体。使用本工具即表示您同意并接受本声明的内容，并愿意承担因使用该功能而产生的风险和责任，如有异议请停止使用本工具。
- 在使用本工具之前，请谨慎阅读并理解以上免责声明的内容。对于使用本工具所产生的任何问题或疑问，请及时联系开发者。

## License

msTT产品的使用许可证，具体请参见[LICENSE](LICENSE)文件。

msTT工具下的文档适用CC-BY 4.0许可证，具体请参见[LICENSE](docs/LICENSE)文件。

## 贡献声明

1. 提交错误报告：如果您在msTT中发现了一个不存在安全问题的漏洞，请在msTT仓库中的Issues中搜索，以防该漏洞被重复提交，如果找不到漏洞可以创建一个新的Issues。如果发现了一个安全问题请不要将其公开，请参阅安全问题处理方式。提交错误报告时应该包含完整信息。
2. 安全问题处理：本项目中对安全问题处理的形式，请通过邮箱通知项目核心人员确认编辑。
3. 解决现有问题：通过查看仓库的Issues列表可以发现需要处理的问题信息，可以尝试解决其中的某个问题。
4. 如何提出新功能：请使用Issues的Feature标签进行标记，我们会定期处理和确认开发。
5. 开始贡献：
   1. Fork本项目的仓库。
   2. Clone到本地。
   3. 创建开发分支。
   4. 本地测试：提交前请通过所有的单元测试，包括新增的测试用例。
   5. 提交代码。
   6. 新建Pull Request。
   7. 代码检视：您需要根据评审意见修改代码，并重新提交更新。此流程可能涉及多轮迭代。
   8. 当您的PR获得足够数量的检视者批准后，Committer会进行最终审核。
   9. 审核和测试通过后，CI会将您的PR合并入到项目的主干分支。

## 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交[Issues](https://gitcode.com/Ascend/mstt/issues)，我们会尽快回复。感谢您的支持。

## 致谢

msTT由华为公司的下列部门联合贡献：

- 昇腾计算MindStudio开发部
- 分布式并行计算实验室
- 华为云昇腾云服务
- 昇腾计算生态使能部
- 2012网络实验室

感谢来自社区的每一个PR，欢迎贡献msTT！
