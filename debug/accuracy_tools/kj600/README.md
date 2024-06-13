# TensorProbe (codename:kj600) 模型训练状态监控工具

## 简介

本项目开发了一个模型训练状态监控工具，能够收集和聚合模型训练过程中的网络层，优化器， 通信算子的中间值，帮助诊断模型训练过程中计算， 通信，优化器各部分出现的异常情况。

## 安装

###  1. 安装依赖

| 依赖软件    |
| ----------- |
| PyTorch     |
| torch_npu   |
| torchvision |

###  2. 安装 kj600

方式一：从 git 直接安装

```
pip install git+https://gitee.com/xiangsen2/kj600.git
```

方式二：下载源码安装

```
git clone https://gitee.com/xiangsen2/kj600.git
cd KJ600
pip install -e .
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
    "mg_direction": true,
    "cc_distribution": {"enable":true, "cc_codeline":[]},
    "alert": {
        "rules": [{"rule_name": "AnomalyTurbulence", "args": {"threshold": 0.5}}]
    },
    "ops": ["min", "max", "norm", "zeros", "id"],
    "eps": 1e-8
}  
```

每个要监控的module有特定的输入输出格式（依赖于模型实现），所以我们需要指定前向输入输出格式和反向计算时输入张量的梯度和输出张量的梯度格式。 如果不清楚的话可以填空字段（"targets":{}），然后将 "print_struct" 字段设置为 true， 之后工具会打印详细的模型结构。 我们也会随时更新更多常用module的格式规范。

下面详细解释各个字段：

"targets"：必选字段，指定需要监控的大模型层， 例如transformer的第0层language_model.encoder.layers.0。如果不清楚模型结构， 可以填空字段（"targets":{}），然后将 "print_struct" 字段设置为 true， 之后监控工具会打印模型中torch module的名字和详细结构，并在第1个step后退出， 你可以从中选择你关心的module。

"input"：可选字段，"tuple[2]:0"的意思是目标module的前向input参数为长度为2的tuple， 我们关心的是tuple第0个元素。

"output"：必选字段，"tensor"的意思是目标module的前向output参数类型为tensor

"input_grad"：可选字段，"tuple[2]:0"的意思是目标module的后向input_grad参数是长度为2的tuple， 我们关心的是tuple的第0个元素。

"output_grad"：必选字段，"tuple[1]:0"的意思是目标module的后向input_grad参数是长度为1的tuple， 我们关心的是tuple的第0个元素。

"module_ranks"：可选字段，用于在分布式训练场景中希望控制在哪些rank开启module监控。如果不填，则默认在所有rank开启。

"print_struct"：可选字段，设置为true后监控工具会打印模型中torch module的名字和详细结构，并在第1个step后退出。不填默认为false。

"ur_distribution": 可选字段，若为true则会统计adam优化器指定模块（targets中指定）参数的update和ratio向量的数值分布，并展示在heatmap里，默认为false。

"xy_distribution": 可选字段， 若为true则会监控指定module（targets中指定）的输入输出张量。 默认为false。

"mv_distribution": 可选字段， 若为true则会监控指定模块中的参数的优化器状态， 默认为false。

"wg_distribution": 可选字段， 若为true则会监控指定模块的参数梯度， 默认为false。 

"alert": 必选字段。 指定自动报警的异常检测机制及其相应的阈值。目前实现的异常检测是AnomalyTurbulence。 如果统计标量超出历史均值的指定浮动范围(threshold指定， 0.5意味着上浮或者下浮50%）。 目前报警是在控制台打印， 未来会实现发邮件和写数据库。 

"mg_direction": 可选字段，若为true则会统计adam优化器的一阶矩（$m_{t-1}$）和当前梯度($g_t$)符号一致的参数比例。

"cc_distribution": 可选字段， 其中“enable”字段控制开关；“cc_codeline”字段指定监控的代码行，如"train.py\\[23\\]"，默认为空列表，不特别指定。"cc_log_only"字段控制是否监控数据,为true时,仅记录调用到的算子及其调用栈。

"ops": 可选字段，与ur_distribution、xy_distribution、mv_distribution、wg_distribution、mg_direction、cc_distribution配合，监控所选张量的min、max、norm、zeros值。其中，zeros代表监控所选张量的元素小于eps的比例，id代表监控所选的非张量本身，默认为[]。

"eps": 可选字段，若ops里包含"zeros"则需要配置，默认为1e-8。

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

   例如在ModelLink/pretrain_gpt.py的model_provider GPTModel构造后加入以下代码：

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
## 高级用法
### 有效秩(有内存和速度问题）
在工具配置文件中加入"params_effrank"："权重矩阵参数名"
"params_effrank": ["language_model.encoder.layers.0.self_attention.query_key_value.weight"]

