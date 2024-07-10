# TensorProbe (codename:kj600) 模型训练状态监控工具

## 简介

本项目开发了一个模型训练状态监控工具，能够收集和聚合模型训练过程中的网络层，优化器， 通信算子的中间值，帮助诊断模型训练过程中计算， 通信，优化器各部分出现的异常情况。

## 安装

###  1. 安装依赖

| 依赖软件        |
|-------------|
| torch       |
| torch_npu   |
| torchvision |
| tensorboard |
| matplotlib  |
| sqlalchemy  |
| pymysql     |

###  2. 安装 kj600

方式一：从 git 直接安装

```
pip install git+https://gitee.com/xiangsen2/kj600.git
```

方式二：下载源码安装

```
git clone https://gitee.com/xiangsen2/kj600.git
cd kj600
pip install .
```

#  快速上手

   下面以Ascend/ModelLink训练框架为例，给出kj600工具的使用方法。

1. 在ModelLink的根目录，创建json配置文件，如llama2_config.json，内容如下：

```
{  
    "targets": {  
        "language_model.encoder.layers.0": {"input": "tuple[2]:0", "output": "tensor", "input_grad":"tuple[2]:0", "output_grad":"tuple[1]:0"}  
    },
    "print_struct": false,
    "module_ranks": [1,2,3,4],
    "ur_distribution": true,
    "xy_distribution": true,
    "mv_distribution": true,
    "wg_distribution": true,
    "cc_distribution": {"enable":true, "cc_codeline":[]},
    "alert": {
        "rules": [{"rule_name": "AnomalyTurbulence", "args": {"threshold": 0.5}}],
        "inform": {"recipient": "database", "connection_str": "mysql+pymysql://username:password@host:port/database"}
    },
    "ops": ["min", "max", "norm", "zeros", "id"],
    "eps": 1e-8
}  
```

每个要监控的module都有自己特定的输入输出格式（依赖于模型实现），所以我们需要指定前向输入输出格式和反向计算时输入张量的梯度和输出张量的梯度格式。 如果不清楚的话可以将"targets"填为空（"targets":{}），然后将 "print_struct" 字段设置为 true， 之后工具会打印详细的模型结构。 我们也会随时更新更多常用module的格式规范。

下面详细解释各个字段：

| 字段名字                                                       | 是否必选    | 解释   | 
| ------------------------------------------------------------ | -------- | -------- | 
|"targets"| 必选 |指定需要监控的大模型层， 例如transformer的第0层language_model.encoder.layers.0。如果不清楚模型结构， 可以将"targets"填为空（"targets":{}），然后将 "print_struct" 字段设置为 true， 之后监控工具会打印模型中torch module的名字和详细结构，并在第1个step后退出， 你可以从中选择你关心的module。|
|"input"| 可选 |"tuple[2]:0"的意思是目标module的前向input参数为长度为2的tuple， 我们关心的是tuple第0个元素。|
|"output"| 必选 |"tensor"的意思是目标module的前向output参数类型为tensor|
|"input_grad"| 可选 |"tuple[2]:0"的意思是目标module的后向input_grad参数是长度为2的tuple， 我们关心的是tuple的第0个元素。|
|"output_grad"| 必选 |"tuple[1]:0"的意思是目标module的后向input_grad参数是长度为1的tuple， 我们关心的是tuple的第0个元素。|
|"print_struct"| 可选 |设置为true后监控工具会打印模型中torch module的名字和详细结构，并在第1个step后退出。不填默认为false。|
|"module_ranks"| 可选 |用于在分布式训练场景中希望控制在哪些rank开启module监控。如果不填，则默认在所有rank开启。|
|"ur_distribution"|  可选 |若为true则会统计adam优化器指定模块（targets中指定）参数的update和ratio向量的数值分布，并展示在heatmap里，默认为false。依赖histc算子， 需要CANN8.0.rc2以上版本， 否则会有严重的性能问题。 |
|"xy_distribution"|  可选 | 若为true则会监控指定module（targets中指定）的输入输出张量。 默认为false。|
|"mv_distribution"|  可选 | 若为true则会监控指定模块中的参数的优化器状态， 默认为false。需要在TrainerMon构造函数正确指定opt_ty. 目前只支持megatron的混合精度优化器以及megatron的分布式优化器。 Deepspeed的分布式优化器实现暂不支持。 |
|"wg_distribution"|  可选 | 若为true则会监控指定模块的参数梯度， 默认为false。 |
|"alert"|  必选 | · "rules": 指定自动报警的异常检测机制及其相应的阈值。目前实现的异常检测是AnomalyTurbulence。 如果统计标量超出历史均值的指定浮动范围(threshold指定， 0.5意味着上浮或者下浮50%）则在控制台打印报警信息。<br>· "inform": 自动报警需要的配置，若想关闭自动报警删掉inform的配置即可。其中"recipient"指定自动报警的通知方式，可选值为"database"或"email"，默认为"database"。<br>- 若"recipient"为"database"，则需要指定"connection_str"字段，即数据库的连接URL，默认为{"recipient":"database", "connection_str": "mysql+pymysql://username:password@host:port/database"}，若有特殊字符需要转义。<br>- 若"recipient"为"email"，则需要指定"send_email_address"-发送方邮箱地址，"receive_email_address"-接收方邮箱地址，"send_email_username"-发送方邮箱用户名，"send_email_password"-发送方邮箱密码，"smtp_server"-发送方邮箱对应的SMTP服务器，"smtp_port"-发送方邮箱对应的SMTP端口号。默认为:<br>{"recipient":"email", send_email_address": "sender@huawei.com", "receive_email_address": "receiver@huawei.com", "send_email_username": "username", "send_email_password": "******", "smtp_server": "smtpscn.huawei.com", "smtp_port": "587"}|
|"cc_distribution"|  可选 | 其中“enable”字段控制开关；需要监控通信算子时，务必尽量早地实例化`TrainerMon`, 因为监控通过劫持原始func后挂hook实现，部分加速库初始化时会保存原始function，避免监控失效。“cc_codeline”字段指定监控的代码行，如:`train.py\\[23\\]`，默认为空列表，不特别指定；"cc_pre_hook"字段控制是否监控通信前的数据； "cc_log_only"为true时,仅记录调用到的算子及其调用栈, 不监控通信的输入输出|
|"ops"|  可选 |与ur_distribution、xy_distribution、mv_distribution、wg_distribution、mg_direction、cc_distribution配合，监控所选张量的min、max、norm、zeros值。其中，zeros代表监控所选张量的元素小于eps的比例，id代表监控所选的非张量本身，默认为[]。|
|"eps"|  可选 |若ops里包含"zeros"则需要配置，默认为1e-8。|

下面给出transformer架构模型中常见的module的前向计算的输入输出和反向计算输入张量的梯度和输出张量的梯度格式，以供参考：

| module                                                       | input    | output   | input_grad | output_grad |
| ------------------------------------------------------------ | -------- | -------- | ---------- | ----------- |
| language_model.embedding.word_embeddings                     | tuple[1] | tensor   | tuple[1]   | tuple[1]    |
| language_model.embedding.embedding_dropout                   | tuple[1] | tensor   | tuple[1]   | tuple[1]    |
| language_model.encoder.layers.0                              | tuple[2] | tensor   | tuple[2]   | tuple[1]    |
| language_model.encoder.layers.0.input_norm                   | tuple[1] | tensor   | tuple[1]   | tuple[1]    |
| language_model.encoder.layers.0.self_attention               | tuple[2] | tuple[2] | tuple[2]   | tuple[2]    |
| language_model.encoder.layers.0.self_attention.query_key_value | tuple[1] | tuple[2] | tuple[1]   | tuple[2]    |
| language_model.encoder.layers.2.self_attention.core_attention_flash | tuple[3] | tensor   | tuple[3]   | tuple[1]    |
| language_model.encoder.layers.0.self_attention.dense         | tuple[1] | tuple[2] | tuple[1]   | tuple[2]    |
| language_model.encoder.layers.0.post_attention_norm          | tuple[1] | tensor   | tuple[1]   | tuple[1]    |
| language_model.encoder.layers.0.mlp                          | tuple[1] | tuple[2] | tuple[1]   | tuple[2]    |
| language_model.encoder.final_norm                            | tuple[1] | tensor   | tuple[1]   | tuple[1]    |

对于language_model.embedding.word_embeddings这类输入层，我们不关心输入的情况下，可以不填"input"和"input_grad"，监控的状态中不会包含输入的相关信息。config文件示例如下：

```
{  
    "targets": {  
        "language_model.embedding.word_embeddings": {"output": "tensor","output_grad":"tuple[1]:0"}  
    }
}  
```

2. 在训练器中加入代码，开启kj600训练监控。

   例如在ModelLink/pretrain_gpt.py的model_provider GPTModel构造后加入以下代码,  **注意优化器类型opt_ty** ：

   ```
       from kj600.module_hook import TrainerMon
       hooker = TrainerMon("./llama2_config.json", params_have_main_grad=True, opt_ty="Megatron_DistributedOptimizer") # or opt_ty=Megatron_Float16OptimizerWithFloat16Params
       hooker.hook_modules(model=model, grad_acc_steps=args.global_batch_size//args.data_parallel_size//args.micro_batch_size) 
   ```
   params_have_main_grad: 若为True则参数权重梯度为main_grad，否则为grad，默认为True。
   
   如果不是Megatron-LM的训练框架， 可以设置对应的梯度累积步数grad_acc_steps。 

   如果要监控混合精度优化器的动量和方差， 需要在混合精度优化器构造后加入如下代码。 目前只支持Megatron_DistributedOptimizer， 使用bf16或者fp16混合精度时开启分布式优化器。 或者Megatron_Float16OptimizerWithFloat16Params， 使用bf16或者fp16混合精度选项并且不开启分布式优化器。 

   ```
   model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
           model_provider, model_type) 
   # 插入位置
   from kj600.module_hook import TrainerMon
   TrainerMon.set_wrapped_optimizer(optimizer)
   ```

3. 配置tensorboard写入的目录

   ```
   export KJ600_OUTPUT_DIR=/xxx/output_dir
   ```

4. 开始预训练，在日志中如果发现以下内容， 则说明指定的模块被成功监视。

   ```
   > language_model.encoder.layers.0 is monitored successfully
   > 1 out of 1 modules are monitored.
   ```

5. 训练过程中，打开tensorboard，可以查看训练的中间状态：

```
tensorboard --logdir=$KJ600_OUTPUT_DIR
```

之后，运行以下SSH命令来建立端口转发，可以在本地通过http://localhost:6006访问tensorboard：

```
ssh -N -L localhost:6006:localhost:6006 your_username@remote_server_address
```

# 高级用法
TBD

# 公开接口

**接口说明**

```python
TrainerMon.__init__(config_file_path, params_have_main_grad=True, opt_ty=None) -> None
```

| 参数  | 说明                  | 是否必选 |
| ----- | -------------------- | -------- |
| config_file_path |自己写的json配置文件路径。 | 是       |
| params_have_main_grad |权重是否使用main_grad，是就为True，否则为False。默认为True。 | 否       |
| opt_ty |优化器类型，有两个选项，Megatron_DistributedOptimizer：使用bf16或者fp16混合精度时开启分布式优化器；Megatron_Float16OptimizerWithFloat16Params：使用bf16或者fp16混合精度选项并且不开启分布式优化器，也适用于常规的adam优化器。如果使用的不是adam优化器，使用None。默认为None。 | 否      |

**接口说明**

```python
TrainerMon.hook_modules(model, grad_acc_steps) -> None
```

| 参数  | 说明                  | 是否必选 |
| ----- | -------------------- | -------- |
| model |需要监控的模型，需要是一个torch.nn.Module。 | 是       |
| grad_acc_steps | 梯度累积步数。 | 是      |

**接口说明**

```python
TrainerMon.set_wrapped_optimizer(_wrapped_optimizer) -> None
```

| 参数  | 说明                  | 是否必选 |
| ----- | -------------------- | -------- |
| _wrapped_optimizer |megatron创建好的混合精度优化器。 | 是       |