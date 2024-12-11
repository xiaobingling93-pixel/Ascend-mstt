# Graph Analyzer

#### 介绍
图分析精度工具

#### 软件架构
软件架构说明


#### 安装教程

1. 下载源码 
```
git clone https://gitee.com/ascend/mstt.git 
```
2.  pip安装
```
cd debug/accuracy_tools/graph_analyzer
pip install .
```

#### 使用说明
- ir图使用推荐：
    - ir图推荐使用`anf_after_graph_build`图

#### 功能说明
**`<>`表示必选参数，`[]`表示可选参数**
1.  ir图结构分析
使用方式：

```
graph_analyzer --ir </path/to/ir/file> [--output </path/to/output/directory>]
```
执行后，会自动分析ir文件，将ir文件分析后的结果输出到指定output目录下的struct.json，如果未指定output则默认为当前目录


2.  数码关联
数码关联是指数据和代码调用栈的关联，数据一般意义上指静态图`O0`,`O1`,`O2`下dump下来的数据
目前支持：
- [x] 全量[tensor(npy)]数据格式的数码关联
- [x] 统计值[statisitc]数据格式的数码关联
- [x] 融合算子场景
- [x] 支持超长算子名dump文件的自动解析
- [x] 反向算子的正向绑定

使用方式:

```
graph_analyzer --ir <path/to/ir/file> --data </path/to/dump/data/dir> [--output </path/to/output/directory>]
```

- 如果是全量模式，则会把数据文件路径和代码调用栈的关联关系存到output路径下的mapping.csv中
- 如果是统计值模式，则会把统计值csv中每个条目加上该条目对应的代码栈
