# 🚨 重要通知

**1. Ascend Training Tools 更名为 MindStudio Training Tools (mstt)。**

**2. 本代码仓 URL 变更为 [https://gitee.com/ascend/mstt](https://gitee.com/ascend/mstt)，原 URL 仍然可用（2024.07.04 ）。**

**3. 不再维护：[api_accuracy_checker](./debug/accuracy_tools/api_accuracy_checker/) （2024.09.30下线）和[ ptdbg_ascend](./debug/accuracy_tools/ptdbg_ascend/)
（2024.09.30下线）**

**相关目录 mstt/debug/accuracy_tools/api_accuracy_checker 和 mstt/debug/accuracy_tools/ptdbg_ascend 将于 2024.09.30 删除。新版本的预检和 ptdbg 已经合到 mstt/debug/accuracy_tools/msprobe 目录下。**

---

# 🧰 MindStudio Training Tools

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Commit Activity](https://img.shields.io/badge/commit%20activity-high-red)
![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)

## [分析迁移工具](https://gitee.com/ascend/mstt/wikis/工具介绍/分析迁移工具/分析迁移工具介绍)

1. [脚本分析工具](https://gitee.com/ascend/mstt/wikis/%E5%B7%A5%E5%85%B7%E4%BB%8B%E7%BB%8D/%E5%88%86%E6%9E%90%E8%BF%81%E7%A7%BB%E5%B7%A5%E5%85%B7/%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7%E4%BD%BF%E7%94%A8%E6%8C%87%E5%AF%BC)

   脚本分析工具可以帮助用户在执行迁移操作前，分析基于 GPU 平台的 PyTorch 训练脚本中算子、三方库套件、API 亲和性以及动态 shape 的支持情况。

2. [（推荐）自动迁移工具](https://gitee.com/ascend/mstt/wikis/%E5%B7%A5%E5%85%B7%E4%BB%8B%E7%BB%8D/%E5%88%86%E6%9E%90%E8%BF%81%E7%A7%BB%E5%B7%A5%E5%85%B7/%E8%87%AA%E5%8A%A8%E8%BF%81%E7%A7%BB%E5%B7%A5%E5%85%B7%E4%BD%BF%E7%94%A8%E6%8C%87%E5%AF%BC)

   自动迁移工具只需在训练脚本中导入库代码即可完成模型脚本的迁移，使用方式简单，且修改内容少。

3. [脚本迁移工具](https://gitee.com/ascend/mstt/wikis/%E5%B7%A5%E5%85%B7%E4%BB%8B%E7%BB%8D/%E5%88%86%E6%9E%90%E8%BF%81%E7%A7%BB%E5%B7%A5%E5%85%B7/%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E5%B7%A5%E5%85%B7%E4%BD%BF%E7%94%A8%E6%8C%87%E5%AF%BC)

   脚本迁移工具通过后端命令行，将 GPU 上训练的 PyTorch 脚本迁移至 NPU 上，得到新的训练脚本用于训练。

4. [训推一体权重转换工具](https://gitee.com/Ascend/mstt/wikis/%E5%B7%A5%E5%85%B7%E4%BB%8B%E7%BB%8D/%E5%88%86%E6%9E%90%E8%BF%81%E7%A7%BB%E5%B7%A5%E5%85%B7/%E8%AE%AD%E6%8E%A8%E4%B8%80%E4%BD%93%E6%9D%83%E9%87%8D%E8%BD%AC%E6%8D%A2%E5%B7%A5%E5%85%B7%E4%BD%BF%E7%94%A8%E6%8C%87%E5%AF%BC)

   训推一体权重转换工具，支持在 GPU 和 NPU 上训练好的模型转成加速推理支持的格式。

## [精度工具](./debug/accuracy_tools/)

[MindStudio Probe（msprobe，MindStudio 精度调试工具）](./debug/accuracy_tools/msprobe)。

## [性能工具](./profiler)

1. [compare_tools（性能比对工具）](./profiler/compare_tools)

   提供 NPU 与 GPU 性能拆解功能以及算子、通信、内存性能的比对功能。

2. [cluster_analyse（集群分析工具）](./profiler/cluster_analyse)

   提供多机多卡的集群分析能力（基于通信域的通信分析和迭代耗时分析）, 当前需要配合 MindStudio Insight 的集群分析功能使用。

3. [advisor](./profiler/advisor)

   将 Ascend PyTorch Profiler 或者 msprof 采集的 PyThon 场景性能数据进行分析，并输出性能调优建议。

## [Tensorboard](./plugins/tensorboard-plugins/tb_plugin)

Tensorboard 支持 NPU 性能数据可视化插件 PyTorch Profiler TensorBoard NPU Plugin。

支持将 Ascend 平台采集、解析的 PyTorch Profiling 数据可视化呈现，也兼容 GPU 数据采集、解析可视化。

## 分支维护策略

1. MindStudio Training Tools 工具版本分支的维护阶段如下：

   | **状态**            | **时间** | **说明**                                         |
   | ------------------- | -------- | ------------------------------------------------ |
   | 计划                | 1—3 个月 | 计划特性                                         |
   | 开发                | 3个月    | 开发特性                                         |
   | 维护                | 6—12个月 | 合入所有已解决的问题并发布版本                   |
   | 无维护              | 0—3 个月 | 合入所有已解决的问题，无专职维护人员，无版本发布 |
   | 生命周期终止（EOL） | N/A      | 分支不再接受任何修改                             |

2. MindStudio Training Tools 分支版本号命名规则如下：

   mstt 仓每年发布 4 个版本，每个版本都将对应一个分支；以 v6.0 为例，其将对应 v6.0.RC1、v6.0.RC2、v6.0.RC3 以及 v6.0.0 四个版本，在仓库中将存在与之对应的分支。

   | **分支**      | **状态** | **发布日期** | **后续状态**               | **EOL日期** |
   | ------------- | -------- | ------------ | ------------------------ | ----------- |
   | **v6.0.0** | 维护     | 2023.12.12   | 预计 2024.12.12 起无维护    |             |
