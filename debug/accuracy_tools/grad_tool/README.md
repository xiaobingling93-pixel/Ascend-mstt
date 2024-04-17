# Ascend模型梯度状态监测工具

训练状态监控工具提供了两种能力

1. 将模型的梯度数据导出。这种功能可以将模型权重的梯度值以统计量的形式采集出来，用以分析问题。
2. 将两份梯度数据进行相似度对比。这种功能可以发现训练过程中问题出现的step，以及抓取反向过程中的问题。

工具支持PyTorch版本：1.11.0/2.0/2.1。

## 工具特性

1. 使用便捷，修改处少
2. 可配置多种过滤项

## 工具安装

1. 将att仓代码下载到本地，并配置环境变量。假设下载后att仓路径为 $ATT_HOME，环境变量应配置为：

   ```bash
   export PYTHONPATH=$PYTHONPATH:$ATT_HOME/debug/accuracy_tools/
   ```

2. 安装依赖pandas、pyyaml

   ```bash
   pip3 install pandas pyyaml tqdm
   ```

## 使用方式

### 梯度数据导出

1. 自己写一个配置文件 config.yaml，样例如下：

      ```python
      level: L2
      param_list: ["module.model.7.bias", "module.model.0.weight"]
      rank: [0, 1, 2, 3]
      step: [0, 1, 2, 3]
      bounds: [-10, -1, -0.1, -0.01, -0.001, 0, 0.001, 0.01, 0.1, 1, 10]
      output_path: /home/pxp1/code/train_test_msft_multi/test/npu_grad_output4
      ```

    **参数解释**

   | 参数名称                       | 说明                                               | 是否必选 |
   |--------------------------------|----------------------------------------------------|----------|
   | level                  | 有三个 level 可选: L0, L1, L2。决定导出数据的程度，level 越大导出数据越详细。      | 是       |
   | param_list             | 一个列表。填写需要导出的梯度数据的变量名称。不指定或列表为空就表示导出所有参数的梯度数据。       | 否       |
   | rank                   | 一个列表。在多卡场景下，填写需要导出梯度数据的卡的rank id，不指定或列表为空就表示导出所有rank的数据。单卡场景不关注这个参数，怎么填都行。 | 否       |
   | step                   | 一个列表，指定需要导出数据的 step。不指定或列表为空就表示导出所有step的数据。| 否 |
   | bounds                 | 一个列表，需要保证由数据小到大排列。用来划分区间以统计值分布。| 否  |
   | output_path            | 输出目录。如果不存在就会创建一个新目录。 | 是 |

    **bounds 和值分布解释**

   + 值分布：梯度数据落在各个区间的元素个数占总元素个数的比例。
   + bounds：一个列表，用来划分出区间以统计值分布。例如传入bounds = [-10, 0, 10]，此时有一个 grad_value: Tensor = [9.3 , 5.4, -1.0, -12.3]，依据 bounds 划分出 (-inf, -10]、(-10, 0]、(0, 10]、(10, inf) 四个区间，然后统计 grad_value 里的数据落在每个区间内的个数，得到 1、1、2、0。如下图所示：

   
   ![Alt text](img/image-1.png)

   **不同级别的 level 的导出数据：**

   | 级别 | 特征数据表头 | 是否有方向数据 |
   |-----| --           | --            |
   | L0  | ("param_name", "MD5", "max", "min", "norm", "shape") | 无 |
   | L1  | ("param_name", "MD5", *intervals, "=0", "max", "min", "norm", "shape") | 无 |
   | L2  | ("param_name", "MD5", "max", "min", "norm", "shape") | 有 |
   | L3  | ("param_name", "MD5", *intervals, "=0", "max", "min", "norm", "shape") | 有 |

   intervals 就是根据值分布划分出的区间

   **方向数据解释：**
   
   因为模型的参数往往非常大，所以存储真实数据是不可接受的，这里折衷一下，只存储梯度数据的正负号（一个布尔值），也就是方向。

   

2. 在模型构造完成时插入如下代码：

      ```python
      from grad_tool.grad_monitor import GradientMonitor
      gm = GradientMonitor("config_path")
      gm.monitor(model)
      ```

   config_path: 传入config.yaml 的路径实例化一个 GradientMonitor 对象
   
   model: 传入刚构造好的模型进行监控


### 输出结果
**输出目录结构**（level 为 L2）
```bash
{output_path}
      ├── rank_{rank_id}
      │        ├── grad_summary_{step}.csv
      │        ├── step_{step}
      │        │        ├── {param_name}.pt
```
+ {timestamp}：梯度工具导出数据的时候会在 output_path 下生成一个时间戳目录，然后在这个时间戳目录下输出结果
+ rank_{rank_id}：在分布式场景下，会记录卡的 rank_id。非分布式场景下，如果是 cpu 则记录进程号，如果是 cpu/gpu 则记录卡号
+ grad_summary_{step}.csv：会分 step 记录每一步的梯度数据统计值
+ step_{step}：这个目录下会存放该 step 的梯度的方向数据
+ {param_name}.pt：模型参数的梯度方向数据

**grad_summary_{step}.csv**

样例

![Alt text](img/image.png)

字段解释

| 字段                  | 含义                                                         |
| --------------------- | ------------------------------------------------------------|
| Param_name            | 模型参数名称。                                                |
| MD5                   | 梯度数据的 MD5 值                                             |
| (-inf, -0.01]...[0.01, inf) | 梯度值落在区间内的元素个数占总元素的比例。               |
| =0                    | 梯度为0的元素个数占总元素的比例。                             |
| Max                   | 最大值                                                       |
| Min                   | 最小值                                                       |
| Norm                  | L2norm值                                                     |
| Shape                 | 形状                                                         |   

### 梯度相似度比对

会根据所导出的权重，分step比对梯度相似度，输出每个权重的梯度相似度和总的梯度相似度。单个权重的梯度相似度为两份方向数据的重合度，总的梯度相似度为每个权重的梯度相似度按元素个数加权.

#### 前提条件

1. 需要两份以相同配置导出的梯度数据。
2. 两份数据导出时都需要将 config.yaml 的 level 设置为 L2 或者 L3，因为比对功能需要方向数据。

#### 使用方式

   1. 单卡比对。新写一个 python 脚本，里面调用 grad_tool.grad_comparator 的 GradComparator.compare 函数，传入的前两个参数分别为梯度数据的 rank 层目录，顺序无所谓，第三个参数为输出目录。如下所示：

      ```python
      from grad_tool.grad_comparator import GradComparator
      GradComparator.compare("需要对比的rank_id级目录",
                             "需要对比的rank_id级目录",
                             "比对结果输出目录")
      ```

   2. 多卡比对。新写一个 python 脚本，里面调用 grad_tool.grad_comparator 的 GradComparator.compare_distributed 函数，传入的前两个参数分别为梯度数据的 rank 层目录，顺序无所谓，第三个参数为输出目录。如下所示：

      ```python
      from grad_tool.grad_comparator import GradComparator
      GradComparator.compare_distributed("配置文件里写的输出目录",
                                         "配置文件里写的输出目录",
                                         "比对结果输出目录")
      ```

### 比对结果

**输出目录结构**(多卡比对结果，单卡则没有 rank_{rank_id} 这一级目录)

```bash
比对结果输出目录
      ├── rank_{rank_id}
      │         ├── similarities.csv
      │         └── similarities_picture
      │                     ├── {param_name}.png
      │                     └── summary_similarities.png
```

**similarities.csv示例**

![Alt text](img/image-2.png)

这份文件记录了所有权重在每一步的梯度相似度和总的梯度相似度

**summary_similarities.png示例**

![Alt text](img/image-3.png)

这是梯度相似度随 step 变化的图片

## 公开接口

```python
GradientMonitor.monitor(model)
```

**参数说明**

| 参数名称    | 说明                                                         | 是否必选 |
| ----------- | ------------------------------------------------------------ | -------- |
| model   | 设置需要监测的模型               | 是       |

```python
GradientMonitor.save_grad(model)
```

**参数说明**

| 参数名称    | 说明                                                         | 是否必选 |
| ----------- | ------------------------------------------------------------ | -------- |
| model   | 设置需要监测的模型               | 是       |

```python
GradientMonitor.__init__(config_path)
```

**参数说明**

| 参数名称    | 说明                                                         | 是否必选 |
| ----------- | ------------------------------------------------------------ | -------- |
| config_path | 配置文件路径，需要以.yaml结尾                                  | 是      |



# FAQ 

