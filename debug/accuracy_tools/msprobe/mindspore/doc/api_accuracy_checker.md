# **MindSpore动态图精度预检工具**

## 简介

MindSpore动态图精度预检工具通过扫描昇腾NPU上用户训练MindSpore模型中所有Mint API，输出精度情况的诊断和分析。工具以模型中所有的Mint API前反向的dump结果为输入，构造相应的API单元测试，将NPU输出与标杆（CPU高精度）比对，计算对应的精度指标，从而找出NPU中存在精度问题的Mint API。


**真实数据/随机生成模式**：精度预检工具支持随机生成模式和真实数据模式，即在预检dump时可以选择由工具构造随机数进行输入获得dump数据或选择获取真实输入数据进行预检dump操作；随机生成模式执行效率高，可以快速获得结果，但数据精度低，只能大致判断精度问题；真实数据模式执行效率略低于随机生成模式，但是数据精度高，可以准确判断精度问题。

**工具支持Mindspore版本**：2.4。

**工具特性**

- 落盘数据小。
- 支持随机生成模式和真实数据模式。
- 单API测试，排除整网中的累计误差问题。

## 预检流程

精度预检操作流程如下：

1. 在NPU和GPU环境下分别安装msprobe工具。详见《[MindStudio精度调试工具](../../README.md)》的“工具安装”章节。
2. 在NPU训练脚本内添加msprobe工具dump接口PrecisionDebugger，采集待预检数据。详见《[精度数据采集](./dump.md)》，注意需要配置level="L1"。
3. 执行预检操作，查看预检结果文件，分析预检不达标的API。

## 预检操作


```bash
msprobe -f pytorch run_ut -api_info ./dump.json
```

| 参数名称                     | 说明                                                         | 是否必选                           |
| ---------------------------- | ------------------------------------------------------------ | ---------------------------------- |
| -api_info或--api_info_file   | 指定API信息文件dump.json。                                   | 是                                 |
| -o或--out_path               | 指定预检结果存盘路径，默认“./”。                        | 否                                 |

预检执行结果包括`accuracy_checking_result_{timestamp}.csv`和`accuracy_checking_details_{timestamp}.csv`两个文件。`accuracy_checking_result_{timestamp}.csv`是API粒度的，标明每个API是否通过测试。建议用户先查看`accuracy_checking_result_{timestamp}.csv`文件，对于其中没有通过测试的或者特定感兴趣的API，根据其API Name字段在`accuracy_checking_details_{timestamp}.csv`中查询其各个输出的达标情况以及比较指标。详细介绍请参见“**预检结果**”。

## 预检结果

精度预检生成的`accuracy_checking_result_{timestamp}.csv`和`accuracy_checking_details_{timestamp}.csv`文件内容详情如下：

`accuracy_checking_details_{timestamp}.csv`

| 字段                | 含义                                                         |
| ------------------- | ------------------------------------------------------------ |
| API Name            | API名称。                                        |
| Bench Dtype         | 标杆数据的API数据类型。                                      |
| Tested Dtype        | 被检验数据的API数据类型。                                  |
| Shape               | API的Shape信息。                                             |
| Cosine              | 被检验数据与标杆数据的余弦相似度。                         |
| MaxAbsErr           | 被检验数据与标杆数据的最大绝对误差。                       |
| MaxRelativeErr      | 被检验数据与标杆数据的最大相对误差。                     |
| Status              | API预检通过状态，pass表示通过测试，error表示未通过。 |
| message             | 提示信息。                                                   |

`accuracy_checking_result_{timestamp}.csv`

| 字段                  | 含义                                                         |
| --------------------- | ------------------------------------------------------------ |
| API Name              | API名称。                                                    |
| Forward Test Success  | 前向API是否通过测试，pass为通过，error为错误。 |
| Backward Test Success | 反向API是否通过测试，pass为通过，error为错误，如果是空白的话代表该API没有反向输出。 |
| Message               | 提示信息。                                                   |

Forward Test Success和Backward Test Success是否通过测试是由`accuracy_checking_details_{timestamp}.csv`中的余弦相似度、最大绝对误差判定结果决定的。具体规则详见“**API预检指标**”。
需要注意的是`accuracy_checking_details_{timestamp}.csv`中可能存在一个API的前向（反向）有多个输出，那么每个输出记录一行，而在`accuracy_checking_result_{timestamp}.csv`中的结果需要该API的所有结果均为pass才能标记为pass，只要存在一个error则标记error。


## API预检指标

API预检指标是通过对`accuracy_checking_details_{timestamp}.csv`中的余弦相似度、最大绝对误差的数值进行判断，得出该API是否符合精度标准的参考指标。详细规则如下：

 - 余弦相似度大于0.99，并且最大绝对误差小于0.0001，标记“pass”，否则标记为“error”。

