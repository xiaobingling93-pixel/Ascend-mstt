# Code Mapping

#### 介绍

数码关联工具

#### 安装教程

请参见[《msprobe 工具安装指南》](../01.installation.md)。

#### 使用说明
- ir图使用推荐：
    - ir图推荐使用`anf_after_graph_build`图

#### 功能说明
**`<>`表示必选参数，`[]`表示可选参数**


1. 数码关联
数码关联是指数据和代码调用栈的关联，数据一般意义上指静态图`O0`,`O1`,`O2`下dump下来的数据
目前支持：
- [x] 全量[tensor(npy)]数据格式的数码关联
- [x] 统计值[statisitc]数据格式的数码关联
- [x] 融合算子场景
- [x] 支持超长算子名dump文件的自动解析
- [x] 反向算子的正向绑定

使用方式:

```
graph_analyzer --ir <path/to/ir/file> --dump_data </path/to/dump/data/dir> [--output </path/to/output/directory>]
```


| 参数名称     | 说明                                                                         |参数类型    | 是否必选     |
| ---------------------------- |----------------------------------------------------------------------------|---------------------- | ---------------------------------- |
| --ir   | 指定 mindspore 静态图运行时生成的ir图。                                                 |  str      | 是      |
| --dump_data    | 指定dump输出路径（支持tensor, statistic)，可以指定父目录也可以指定文件。                            |   str   | 是      |
| --output | 默认为"./"，只在tensor模式时生效，会把数据文件路径和代码调用栈的关联关系存到output路径下的code_mapping{时间戳}.csv中。 |  str   | 否      |


- 如果是statistic模式，则会把统计值csv中每个条目加上该条目对应的代码栈。
