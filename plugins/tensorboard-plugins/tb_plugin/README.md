# PyTorch Profiler TensorBoard NPU Plugin

### 下架说明
本插件代码已不再维护，推荐使用[MindStudio Insight](https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/)工具获取更为强大的调优体验。

### 介绍
此工具是PyTorch profiling数据以及可视化的TensorBoard的插件。 \
它支持将Ascend平台采集、解析的Pytorch Profiling数据可视化呈现，也兼容GPU数据采集、解析可视化，支持PyTorch 2.0GPU版本的profiling数据可视化。同时集成了精度比对的功能，支持查看loss曲线和比对两个网络的loss收敛趋势。

### 快速安装说明
* 相关依赖：
  pandas >= 1.0.0 ，tensorboard >= 2.11.0，protobuf <= 3.20.3
* 安装方式
    * 本插件已经上传到pypi社区，用户可在python环境下直接通过以下pip指令进行安装：\
    `pip install torch-tb-profiler-ascend`
    * 也可在pypi社区上下载离线whl包，传输到无法访问公网的环境上离线安装使用。访问[下载链接](https://pypi.org/project/torch-tb-profiler-ascend/#files)选择whl包进行下载，之后便可使用指令安装（此处{version}为whl包实际版本）\
    `pip install torch-tb-profiler_ascend_{version}_py3_none_any.whl`

### 解析数据说明

* 准备profiling数据

  需要在读取的目录下放置指定格式的profiling数据。格式为包含3个层级的目录结构：runs层级为最外层目录（我们将一个完整的Profiling数据视为一个runs进行可视化处理），其子目录为worker_span层级（命名格式为{worker}_{span}_ascend_pt，<font color='red'>注：此处span为数字，代表时间戳</font>），下一层级为规定命名的ASCEND_PROFILER_OUTPUT目录，此目录中包含此插件加载展示的数据文件，如trace_view.json、kernel_details.csv、operator_details.csv等。
  目录结构如下：
*  E.g. there are 2 runs: run1, run2 \
            `run1` \
                `--[worker1]_[span1]_ascend_pt` \
                    `----ASCEND_PROFILER_OUTPUT` \
                        `------trace_view.json` \
                        `------kernel_details.csv` \
                `--[worker2]_[span1]_ascend_pt` \
                    `----ASCEND_PROFILER_OUTPUT` \
                        `------trace_view.json` \
                        `------operator_details.csv` \
            `run2` \
                `--[worker1]_[span1]_ascend_pt` \
                    `----ASCEND_PROFILER_OUTPUT` \
                        `------memory_record.csv` \
                        `------operator_memory.csv`

### 启动方式
  
1. 启动TensorBoard

  `tensorboard --logdir=./samples`

  如果网络浏览器与启动TensorBoard的机器不在同一台机器上，则需要在尾部加上`--bind_all`命令，如：

  `tensorboard --logdir=./samples --bind_all`

  注意：确保默认端口6006对浏览器的主机打开。

  如果需要切换端口号需要在尾部加上指定的端口号，如`--port=6007`

  `tensorboard --logdir=./samples --port=6007`

2. 在浏览器上打开tensorboard

  在浏览器中打开URL： `http://localhost:6006`。
  如果tensorboard启动命令使用`--bind_all` , 主机名不是`localhost`，而是绑定的主机ip，可以在cmd之后打印的日志中查找。

  注意：如果`--logdir` 指定目录下的文件太大或太多，请等候，刷新浏览器查看加载结果。

### PyTorch Profiling
#### 页面展示说明

  页面加载完成后，左侧视图如图。每个Runs都对应于`--logdir`指定的文件夹下的一个子文件夹（三层目录中的第一层run1, run2等）。
  每个子文件夹包含一个或多个profiling数据文件夹。

  ![Alt text](./docs/images/control_panel.PNG)

  Runs: `--logdir`下包含三层目录的所有数据。

  Views: 展示数据分析的多个视图，包含Operator、NPU Kernel、Trace、Memory等多个view。

  Workers-Spans: 多线程的情况下Profiling可能包含多组数据，通过Workers和Spans下拉框来选择不同线程和不同时间采集的数据产生的结果。

##### Operator View

  Operator View展示的是运行在host侧和device侧的Pytorch算子、计算算子的详细信息。

  ![Alt text](./docs/images/operator_view.PNG)

  Calls: 表示的是运行过程中此算子被调用的次数。
  
  Input Shapes: shapes信息。

  Device Self Duration: 算子在device侧的耗时（除去子算子）。

  Device Total Duration: 算子在device侧的耗时。

  Host Self Duration: 算子在host侧的耗时（除去子算子）。

  Host Total Duration: 算子在host侧的耗时。

  AI Cores Eligible: 此算子是否在AICore上运行。

  AI Cores Self (%): 算子在AICore上的耗时（除去子算子） / Device Self Duration。

  AI Cores Total (%):  算子在AICore上的耗时 / Device Total Duration。

  CallStack:  此算子的所有调用堆栈信息。

  说明: 由于一些算子之间存在父子关系（在trace上显示为包含关系），Self表示除去子算子的耗时，Total表示包含所有子算子的耗时。

  ![Alt text](./docs/images/vscode_stack.PNG)

  页面展示了四个饼图和两张表，通过界面的Group By切换表格和饼图。当切换为Operator时，表格以算子名称的维度进行展示，点击某个算子的View CallStack后，此算子会按照Call Stack分类展示算子信息。点击View call frames可以查看算子的调用信息。
  当Group By切换为Operator + Input Shape时，算子以name和Input Shape为维度进行展示。

  ![Alt text](./docs/images/operator_view_group_by_inputshape.PNG)

##### Kernel View

  Kernel View 展示算子在加速核上运行的详细信息。此视图包含两张饼图和两张表，可通过 Group By 切换表格数据：算子的详情表以及统计表。

  * 上方为饼图，展示耗时最多的数个算子耗时比例信息（左侧饼图）和算子执行在各类加速核上耗时百分比（右侧饼图）

  ![Alt text](./docs/images/kernel_view.PNG)

  * 选择 Group By 为 All 时，展示算子详情表，部分字段说明如下：

  | 字段名           | 说明                                   |
  | ---------------- | -------------------------------------- |
  | Step Id          | 标识在哪个 Step 采集的数据             |
  | Name             | 运行在 npu 上的算子名称                |
  | Type             | 算子类型                               |
  | Accelerator Core | AI 加速核类型，包括 AI Core、AI CPU 等 |
  | Start Time(us)   | 算子执行开始时间                       |
  | Duration(us)     | 当前算子执行耗时                     |
  | Wait Time(us)    | 算子执行等待时间                       |
  | Block Dim        | 运行切分数量，对应任务执行时的核数     |

  ![Alt text](./docs/images/kernel_view_group_by_statistic.PNG)

  * 选择 Group By 为 Statistic 时，展示算子信息统计表，此表格展示各算子的执行统计信息，字段说明如下：

  | 字段名           | 说明  |
  | ---------------- | -------|
  | Name             | 运行在 npu 上的算子名称 |
  | Calls            | 算子执行次数 |
  | Total Duration(us) | 算子执行总时间 |
  | Min Duration(us)  | 算子执行的最小时间 |
  | Max Duration(us)  | 算子执行的最大时间 |
  | Avg Duration(us)  | 算子执行平均时间  |

##### Trace View

  此视图显示使用chrome插件，展示在整个训练过程中的时序图。

  ![Alt text](./docs/images/trace_view.PNG)

  Trace View主要包含三个层级以及层级下各个线程上执行的算子的时序排布。

  ![Alt text](./docs/images/trace_view_one_step.PNG)

  目前主要包括三个层级，PTA、CANN和Ascend Hardware。可以通过选择Processes来选择要展示的层级。

  ![Alt text](./docs/images/trace_view_launch.PNG)

  选择只展示async_npu，可以查看框架侧算子与昇腾硬件上执行的算子的下发执行关系。

  ![Alt text](./docs/images/trace_view_npu_utilization.PNG)

  ![Alt text](./docs/images/trace_view_fwd_bwd_correlation.PNG)

  <font color='red'>Tips：通过键盘的'W/S'键可以以光标位置为中心放大/缩小当前区域，通过'A/D'可以左移/右移当前可视域。</font>

##### Memory View

  展示的是Pytorch Profiler执行过程中内存申请和释放的信息。
  主要包括两张折线图和两张表。可以在 'Device' 下拉框下选择要展示的NPU卡的内存使用信息。Group By可以切换总的内存使用和各个组件内存使用图表。

  * Operator

    整个采集过程中，算子内存使用情况汇总。

    ![Alt text](./docs/images/memory_view.PNG)
    表格数据代表含义:

    * Name: 算子名称。

    * Size: 申请的内存大小。

    * Allocation Time: 内存申请时间。

    * Release Time: 内存释放时间。

    * Duration: 内存持有时间。

  * Component

    折线图为算子级上报的PTA侧和GE侧的内存持有和实际使用信息，以及进程级内存申请的趋势变化。表格为组件级内存峰值信息表，展示各NPU组件的内存峰值以及达到峰值的时刻。
    
    ![Alt text](./docs/images/memory_view_component.PNG)
    表格数据代表含义:

    * Component: 组件名称。

    * Peak Memory Reserved: 组件内存持有峰值。

    * Time: 达到内存峰值的时刻（若存在多个相同峰值则取首次达到峰值时刻）。

##### Diff View

  Diff视图提供了Profiling数据比对功能。适用于同一网络不同迭代之间采集数据比对算子耗时情况，网络进行优化前后相同位置算子耗时情况比对、单机多卡不同卡之间采集数据比对以及相同网络不同硬件平台上运行性能情况比对等场景。
  ![Alt text](./docs/images/diff_view.png)
  
  * 最上方为整体比对，以采集的step为周期比较两份数据各类算子的耗时情况以及累计耗时变化趋势。点击其中某块柱形，可以单点查看对应详情。
  
    ![Alt text](./docs/images/diff_detail.png)

  * 中间视图为差异图，由红蓝两块区域构成。横坐标与上方视图对应，蓝色区域为每类算子的耗时差值，红色区域表示当前所有算子耗时差的累加值。

  * 最下方为算子比对明细表，显示相关差值以及相差比例信息。由于数据条目较多，支持选择是否显示Host Duration、Self Host Duration、Device Duration以及Self Device Duration相关比对信息。
      * Host Duration：算子在Host侧的累计耗时，包括子算子耗时。
      * Self Host Duration：算子在Host侧的累计耗时，不包括子算子耗时。
      * Device Duration：算子在Device侧的累计耗时，包括子算子耗时。
      * Self Device Duration：算子在Device侧的累计耗时，不包括子算子耗时。

##### Distributed View

  Distributed视图展示的是多卡采集数据情况，包括每张卡的计算、通信信息以及通信算子的详细信息，界面由两张柱状图和一个通信算子信息表构成，如下图。
  ![Alt text](./docs/images/distributed_view.PNG)
  
  * 左侧柱状图呈现了每张卡计算和通信等项的耗时，各项定义如下：

  | 字段 | 含义 |
  |------|------|
  | Computation   | 计算时间：在NPU上的计算时间减去和通信重叠的时间。|
  | Communication | 通信时间：总通讯时间减去和计算重叠的时间。|
  | Overlapp      | 重叠时间：计算和通信重叠的时间。此项占比越大代表计算和通信的并行性越好，理想情况下计算和通信完全重叠。|
  | Other         | 除去计算和通信的其他部分耗时，包括初始化、数据加载等。|
  
  * 右侧柱状图将通信时间分为数据传输时间和同步时间进行统计，定义如下：
  
  | 字段 | 含义 |
  |------|------|
  | Data Transfer Time | 通信时间中实际的数据传输时间。     |
  | Synchronizing Time | 通信时间中等待以及和其他卡同步的时间。 | 

  * 界面下方为通信算子信息表，统计了各张卡的通信算子详情。

  | 字段 | 含义 |
  |------|------|
  | Name | 通信算子名称       |
  | Calls | 调用次数。        |
  | Total Transit Size(bytes) | 传输的总数据大小。    |
  | Avg Transit Size(bytes) | 平均每次传输的数据大小。 |
  | Elapse Time(us) | 此类算子总耗时。     |
  | Avg Elapse Time(us) | 单个算子平均耗时。    |
  | Transit Time(us) | 此类算子传输总耗时。   |
  | Avg Transit Time(us) | 单个算子平均传输耗时。  |

### Loss Comparison
#### 工具介绍

  Loss Comparison是集成在该插件上的精度比对工具，提供了对loss曲线的可视化，loss数据匹配导出csv，以及两份数据比对等功能。

#### 页面展示说明
  切换顶部页签栏至ACCURACY页签，即可进入精度比对工具页面。

##### 文件配置
###### 文件导入
  界面分为左侧边栏和右侧展示界面。点击左侧的Import Files或在左侧未勾选文件时点击右侧界面中心的Import Files字体，将会弹出系统文件资源管理窗，可以上传需要比对的模型网络训练日志文件。

  **注：当前最多支持上传6个文件，单个文件大小不能超过50MB。**
  ![Alt text](./docs/images/accuracy.PNG)

###### 已上传文件操作
  文件上传后，在左侧侧边栏出现文件列表。每个文件栏内都有配置数据匹配条件、导出CSV以及删除三种操作图标。

  ![Alt text](./docs/images/accuracy_file_operator.PNG)

  * 点击配置数据匹配条件图标后，出现匹配条件配置弹框，需要设置Loss Tag和Iteration Tag两个配置项，弹框内每个Tag都包含一个输入框。
  ![Alt text](./docs/images/accuracy_config_modal.PNG) 
  根据2个Tag的取值有如下3点匹配规则：
    1. 匹配数据时将逐行读取文件，查找是否存在输入框内设定的文本，若找到该文本，若为Loss Tag则查找其后是否存在数字或以科学计数法表示的数字（忽略两者中间空格），若为Iteration Tag则查找其后是否存在整数（忽略两者中间空格）。
    2. 若存在多个匹配项，将第一项作为匹配值。
    3. 只有当Loss Tag和Iteration都存在匹配值时，该行的Iteration和Loss才会为有效数据。
    
    E.g.

    ![Alt text](./docs/images/accuracy_file.PNG)

    对于以上这份txt文件，当设定Loss Tag为`loss:`以及Iteration Tag为`iteration`时：
       * 根据上方第1点规则，Iteration Tag可匹配图中区域1内的整数，但无法匹配区域3内的整数，因为`iteration`和整数中间多了非数字字符`:`。
       * Loss Tag可匹配图中区域2和4内的数字，但区域2内为第一项匹配值，根据上方第2点规则，因此只取区域2内数字。
       * Loss Tag在图中区域5内有匹配数据，Iteration Tag在图中区域6内有匹配数据，但由于Iteration Tag在区域5内没有匹配数据，Loss Tag在图中区域6内没有匹配数据，根据上方第3点规则，区域5和区域6内不存在有效数据。
    
    因此上方这张图中最终提取出的有效数据为区域1和区域2内的同一行数字的集合。

  * 点击导出CSV图标后，将导出找到的Iteration和Loss数据为csv文件。 \
  ![Alt text](./docs/images/accuracy_csv.PNG)

  * 点击删除图标后，界面弹出确认删除框，确认后可移除该文件。
  ![Alt text](./docs/images/accuracy_delete.PNG)

##### Loss数据看板
  已上传文件后，可在左侧侧边栏勾选该文件，右侧则会展示该文件的Loss数据看板，包含loss折线图和明细表格。
  
  * 勾选单个文件时，Loss数据看板将会占满整个右侧展示界面。
  ![Alt text](./docs/images/accuracy_single_file.PNG)

  * 勾选两个以上文件时，右侧将会展示Loss数据看板和Loss比对看板。
  ![Alt text](./docs/images/accuracy_multiple_files.PNG)

  * Loss数据看板为全量展示，折线图内展示的是所有勾选文件的所有数据，表格内展示的同样为勾选文件的全量数据，若表格内iteration为某些文件独有，则其他文件该行显示为`NA`。

##### Loss比对看板
  当勾选文件为2个以上时，将展示Loss比对看板，Loss比对看板基于iteration取两份比对数据的交集进行展示。
  
  * 在Comparison objects中选择两个文件，则展示该两个文件的比对信息。
  ![Alt text](./docs/images/accuracy_loss_comparison.png)

  * 比对方式有三种，通过Comparison Setting进行设定。
    * Comparison Normal：相同iteration，后选择文件的loss值减去先选择文件的loss值。
    * Comparison Absolute：相同iteration，两个文件的loss的差值的绝对值。
    * Comparison Relative：相同iteration，两个文件的loss的差值的绝对值 / 先选择文件的loss值。

### 公网URL说明

[公网URL说明](./docs/公网URL说明.xlsx)