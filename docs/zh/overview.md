# 简介

MindStudio Training Tools（MindStudio训练工具链，msTT）聚焦您在模型迁移、模型开发中遇到的痛点问题，提供全流程的工具链，通过提供分析迁移工具、精度调试工具、性能调优工具三大主力工具包，帮助您解决开发过程中迁移困难、Loss跑飞、性能不达标或劣化等问题，让您轻松解决精度和性能问题，开启乐趣十足的极简开发之旅。

**模型训练开发全流程**

![training_process](figures/training_process.png)

# 功能介绍

## 分析迁移工具

MindStudio Analysis and Migration Tool（MindStudio分析迁移工具，msfmktransplt）

   PyTorch训练脚本一键式迁移至昇腾NPU的功能，开发者可做到少量代码修改或零代码完成迁移。详细介绍请参见《[分析迁移工具](https://gitcode.com/Ascend/mstt/blob/master/msfmktransplt/docs/zh/msfmktransplt_instruct.md)》。

## 精度调试工具

MindStudio Probe（MindStudio精度调试工具，msProbe）

模型开发精度调试环节使用的工具包，是针对昇腾提供的全场景精度工具链，帮助用户提高模型精度问题定位效率。详细介绍请参见《[精度调试工具](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/dump/mindspore_dump_quick_start.md)》。

## 性能调优工具

- MindStudio Profiler（MindStudio模型调优工具，msProf）

  构建昇腾全场景性能调优基础能力，支持采集CANN和NPU性能数据，提升昇腾设备性能调优效率。详细介绍请参见《[模型调优工具](https://gitcode.com/Ascend/msprof/blob/master/docs/zh/getting_started/quick_start.md)》。

- Ascend PyTorch调优工具

  提供PyTorch训练/在线推理场景采集性能数据，输出可视化的性能数据文件，提升性能分析效率。详细介绍请参见《[Ascend PyTorch调优工具](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md)》。

- MindStudio Profiler Analyze（MindStudio性能分析工具，msprof-analyze）

  昇腾性能分析工具，基于采集的性能数据进行分析，提供昇腾设备性能瓶颈快速识别能力。详细介绍请参见《[性能分析工具](https://gitcode.com/Ascend/msprof-analyze/blob/master/docs/zh/getting_started/quick_start.md)》。

- msMemScope（MindStudio内存分析工具）

  针对昇腾显存调试调优场景的专用工具，提供整网级多维度显存数据采集、自动诊断、优化分析能力。详细介绍请参见《[内存分析工具](https://gitcode.com/Ascend/msmemscope/blob/master/docs/zh/quick_start.md)》。

- MindStudio Monitor（MindStudio在线监控工具，msMonitor）

  一站式在线监控工具，支持落盘和在线性能数据采集，提供集群场景性能监测及定位能力。详细介绍请参见《[在线监控工具](https://gitcode.com/Ascend/msmonitor/blob/master/docs/zh/getting_started/quick_start.md)》。

- MindStudio Profiling Tools Interface（msPTI）

  MindStudio针对Ascend设备提出的一套Profiling API，用户可以通过msPTI构建针对NPU应用程序的工具，用于分析应用程序的性能。详细介绍请参见《[msPTI工具](https://gitcode.com/Ascend/mspti/blob/master/docs/zh/getting_started/quick_start.md)》。

- MindStudio Insight（MindStudio可视化调优工具，msInsight）

  MindStudio Insight可视化工具，支持系统级、算子级、服务化等多场景多维度性能分析，深度剖析性能数据，帮助开发者完成性能诊断。详细介绍请参见《[可视化调优工具](https://gitcode.com/Ascend/msinsight/blob/master/docs/zh/user_guide/overview.md)》。

- 昇腾亲和性CPU绑核工具

  绑核脚本，支持非侵入修改工程代码，实现一键式绑核功能。详细介绍请参见《[昇腾亲和性CPU绑核工具](https://gitcode.com/Ascend/mstt/blob/master/profiler/affinity_cpu_bind/README.md)》。

- Tinker

  Tinker大模型并行策略自动寻优系统，根据提供的训练脚本，进行单节点NPU性能测量，推荐高性能并行策略训练脚本。详细介绍请参见《[Tinker](https://gitcode.com/Ascend/mstt/blob/master/profiler/tinker/README.md)》。
