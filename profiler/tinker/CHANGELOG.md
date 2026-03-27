# Profiler

1.3.3

- 子图前向代码动态提取

1.3.1b

- 根据脚本动态生成profile范围
- 校正子图wrap逻辑，自动化生成attention_mask

1.3.0

- 适配ModelLink 1.0.RC3(对应袁明明方式的ModelLink-1.2)
- 使用block adapter重构mcore block

1.2.2A (adapter)

- 使用adapter封装对训练框架的import
- 4block: 使用block adapter重构legacy block 

1.2.2

- 张量信息自动提取
- 在`barrier`之后再多跑一轮预热，并将`event.start()`放在预热后
- 将`host`侧`time.time()`时间测量方式修改为`torch.cuda.Event`的`elapsed_time`测量方式，去除所有`synchronize`操作
- 在时间测量前新增`barrier`同步各设备

1.2.0

- 适配ModelLink 1.0.RC2(对应原命名方式的ModelLink-1.1)

# Optimizer

1.3.0

- 入口改造
- 修复并行策略重复问题

1.2.0

- 调整`dist_opt`切分内容：2阶优化器状态、全精度权重参数
- 调整`reserved`内存仿真逻辑
- 增加`attention_mask`内存占用仿真
- 调整`recompute`内存仿真，开启时保留`input`内存占用
