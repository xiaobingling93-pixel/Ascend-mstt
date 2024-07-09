# 精度工具

MindStudio Training Tools工具针对模型训练精度问题设计推出了一系列精度工具，包括模型精度预检工具和PyTorch精度工具的精度比对功能、溢出检测功能、通信精度检测等功能。这些工具有各自侧重的场景，用于辅助用户定位模型精度问题。

### 子功能介绍

NPU上训练的网络存在精度问题，精度指标（loss或者具体的评价指标）与标杆相差较多。对于该场景的问题，可以使用**Ascend模型精度预检工具**或者**PyTorch精度工具**进行定位。

| 工具名称                                                     | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [api_accuracy_checker（Ascend模型精度预检工具）](https://gitee.com/ascend/mstt/tree/master/debug/accuracy_tools/api_accuracy_checker) | 在昇腾NPU上扫描用户训练模型中所有API，进行API复现，给出精度情况的诊断和分析。 |
| [ptdbg_ascend（PyTorch精度工具）](https://gitee.com/ascend/mstt/tree/master/debug/accuracy_tools/ptdbg_ascend) | 进行PyTorch整网API粒度的数据dump、精度比对和溢出检测，从而定位PyTorch训练场景下的精度问题。 |

### 场景介绍

**Ascend模型精度预检工具**会对全网每一个API根据其实际训练中的shape、dtype和数值范围生成随机的输入，对比它与标杆的输出差异，并指出输出差异过大不符合精度标准的API。该工具检查单API精度问题准确率超过80%，对比一般dump比对方法减少落盘数据量99%以上。具体使用请参见《[Ascend模型精度预检工具](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/api_accuracy_checker/README.md)》

**PyTorch精度工具精度比对功能**可以对NPU整网API数据进行与CPU或GPU标杆数据的精度比对，从而检测精度问题。具体来说，dump统计量、分段dump、模块化dump，通讯算子dump等功能可以用较轻的数据量实现不同侧重的精度比对，从而定位精度问题。具体使用请参见《[ptdbg_ascend精度工具功能说明](https://gitee.com/ascend/mstt/tree/master/debug/accuracy_tools/ptdbg_ascend/doc)》。

**PyTorch精度工具溢出检测功能**是在判断训练网络可能存在溢出现象时，例如某个step的loss突然变成inf nan，或者混精场景下loss_scale不断减小，可以通过ptdbg_ascend的精度检测工具检测API的溢出情况。具体使用请参见《[ptdbg_ascend精度工具功能说明](https://gitee.com/ascend/mstt/tree/master/debug/accuracy_tools/ptdbg_ascend/doc)》。