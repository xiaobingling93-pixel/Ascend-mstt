<h1 align="center">MindStudio Training Tools</h1>

<div align="center">
<h2>昇腾 AI 训练开发工具链</h2>

 [![Ascend](https://img.shields.io/badge/Community-MindStudio-blue.svg)](https://www.hiascend.com/developer/software/mindstudio) 
 [![License](https://badgen.net/badge/License/MulanPSL-2.0/blue)](./LICENSE)

</div>

## ✨ 最新消息

<span style="font-size:14px;">

🔹 **[2026.03.28]**：精度调试模块（debug 目录）正式日落下线，详情请参见 [公告](https://gitcode.com/Ascend/mstt/discussions/2)               
🔹 **[2026.02.25]**：Tinker 并行策略自动寻优系统正式开源，详情请参见 [Tinker 项目](https://gitcode.com/Ascend/tinker)      
🔹 **[2026.01.12]**：本仓库许可证（License）变更，详情请参见 [公告](https://gitcode.com/Ascend/mstt/discussions/1)     
🔹 **[2025.12.31]**：MindStudio 训练开发工具链全面开源    

</span>

## ℹ️ 简介

MindStudio Training Tools（msTT）训练开发工具链，聚焦训练开发中的关键挑战。通过提供分析迁移、精度调试与性能调优三大核心工具，高效应对迁移受阻、Loss 异常、性能不达标等问题，助力实现精度与性能双优的极简开发体验。

<img src="./docs/zh/figures/readme/fullview.svg?v=2026033001" width="1200"/>

## ⚙️ 功能介绍

训练开发工具链提供以下系列化工具：

| 类别 | 工具名称                                                                                      | 功能简介                                               |
|:--:|:------------------------------------------------------------------------------------------|:---------------------------------------------------|
| 迁移 | [**msTransplant**](./msfmktransplt)                                                       | **【分析迁移】** PyTorch 训练脚本一键迁移至昇腾 NPU，支持少量改码或零改码完成迁移。 |
| 精度 | [**msProbe**](https://gitcode.com/Ascend/msprobe)                                         | **【精度调试】** 昇腾全场景精度工具，用于训练精度调试与问题定位。                |
| 精度 | [**TensorBoard**](https://gitcode.com/Ascend/msprobe/tree/master/plugins/tb_graph_ascend) | **【分级可视】** 分级展示模型结构与精度，支持调试与标杆模型对比以定位精度问题。         |
| 性能 | [**msProf**](https://gitcode.com/Ascend/msprof)                                           | **【模型调优】** 全场景性能调优底座，采集 CANN 与 NPU 数据，提升设备调优效率。    |
| 性能 | [**msprof-analyze**](https://gitcode.com/Ascend/msprof-analyze)                           | **【性能分析】** 基于采集数据做性能分析，快速识别性能瓶颈。                   |
| 性能 | [**msMemScope**](https://gitcode.com/Ascend/msmemscope)                                   | **【内存调优】** 内存调优专用工具：整网级多维度内存采集，支持自动诊断与优化分析。        |
| 性能 | [**msInsight**](https://gitcode.com/Ascend/msinsight)                                     | **【可视调优】** 可视化性能分析，覆盖系统、算子、服务化等场景，辅助完成性能诊断。        |
| 性能 | [**Tinker**](https://gitcode.com/Ascend/mstt/tree/master/profiler/tinker)                 | **【并行寻优】** 大模型并行策略自动寻优：按训练脚本做单节点 NPU 测评并推荐高性能并行方案。 |
| 性能 | [**bind_core**](https://gitcode.com/Ascend/mstt/tree/master/profiler/affinity_cpu_bind)   | **【一键绑核】** CPU 绑核工具，无需侵入修改工程即可按 CPU 亲和性策略绑核。       |
| 性能 | [**msPTI**](https://gitcode.com/Ascend/mspti)                                             | **【性能剖析】** 面向昇腾的 Profiling API，可据此开发 NPU 应用性能分析工具。 |
| 监控 | [**msMonitor**](https://gitcode.com/Ascend/msmonitor)                                     | **【在线监控】** 一站式监控，支持落盘与在线采集，面向集群的监测与问题定位。           |

## 🚀 快速入门

面向 PyTorch 与 MindSpore 场景，通过可执行样例串联迁移分析、精度调试与性能调优，助力用户快速上手端到端训练优化。

| 训练框架      | 快速入门指南 |
|-----------|----------------|
| PyTorch   | [《PyTorch 场景 msTT 工具快速入门》](docs/zh/quick_start/pytorch_mstt_quick_start.md) |
| MindSpore | [《MindSpore 场景 msTT 工具快速入门》](docs/zh/quick_start/mindspore_mstt_quick_start.md) |

## 📦 安装指南

介绍 msTT 工具的环境依赖与安装方法，请参见 [《msTT 安装指南》](./docs/zh/install_guide/mstt_install_guide.md)。

## 📘 使用指南

各工具的详细使用说明请参阅其源码仓库中的 README 文件，也可通过上方功能介绍表格中的链接直接跳转。

## 🛠️ 贡献指南

欢迎参与项目贡献，请参见 [《贡献指南》](./docs/zh/contributing/contributing_guide.md)。

## ⚖️ 相关说明

🔹 [《版本说明》](./docs/zh/release_notes/release_notes.md)    
🔹 [《许可证声明》](./docs/zh/legal/license_notice.md)     
🔹 [《安全声明》](./docs/zh/legal/security_statement.md)     
🔹 [《免责声明》](./docs/zh/legal/disclaimer.md)    

## 🤝 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交 [Issues](https://gitcode.com/Ascend/mstt/issues)，我们会尽快回复。感谢您的支持。

|                                      📱 关注 MindStudio 公众号                                       | 💬 更多交流与支持                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|:-----------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <img src="./docs/zh/figures/readme/officialAccount.png" width="120"><br><sub>*扫码关注获取最新动态*</sub> | 💡 **加入微信交流群**：<br>关注公众号，回复“交流群”即可获取入群二维码。<br><br>🛠️ **其他渠道**：<br>👉 昇腾助手：[![WeChat](https://img.shields.io/badge/WeChat-07C160?style=flat-square&logo=wechat&logoColor=white)](https://gitcode.com/Ascend/mstt/blob/master/docs/zh/figures/readme/xiaozhushou.png)<br>👉 昇腾论坛：[![Website](https://img.shields.io/badge/Website-%231e37ff?style=flat-square&logo=RSS&logoColor=white)](https://www.hiascend.com/forum/) |

## 🙏 致谢

msTT 由华为公司的下列部门联合贡献：    
🔹 昇腾计算MindStudio开发部  
🔹 昇腾计算生态使能部  
🔹 华为云昇腾云服务  
🔹 2012分布式并行计算实验室  
🔹 2012网络技术实验室  
感谢来自社区的每一个 PR，欢迎贡献 msTT！
