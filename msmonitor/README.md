# msMonitor: MindStudio一站式在线监控工具

## 安装方式

### 下载包安装

1. 压缩包下载

| msmonitor版本 | 发布日期       | 下载链接   | 校验码  |
|------------|------------|-------------------------------------------------------------------------------------------------------------------------------------------------| ------------------------------------------------------------ |
| 8.1.0       | 2025-07-11 | [aarch64_8.1.0.zip](https://ptdbg.obs.cn-north-4.myhuaweicloud.com/profiler/msmonitor/8.1.0/aarch64_8.1.0.zip)             | ce136120c0288291cc0a7803b1efc8c8416c6105e9d54c17ccf2e2510869fada |
|             | 2025-07-11 | [x86_8.1.0.zip](https://ptdbg.obs.cn-north-4.myhuaweicloud.com/profiler/msmonitor/8.1.0/x86_8.1.0.zip)             | 097d11c7994793b6389b19259269ceb3b6b7ac5ed77da3949b3f09da2103b7f2 |

2. 包校验。

   1. 根据以上下载链接下载包到Linux安装环境。

   2. 进入zip包所在目录，执行如下命令。

      ```
      sha256sum {name}.zip
      ```

      {name}为zip包名称。

      若回显呈现对应版本zip包一致的**校验码**，则表示下载了正确的性能工具zip安装包。示例如下：

      ```bash
      sha256sum aarch64_8.1.0.zip
      ```

3. 包安装(以x86为例)

   1. 解压压缩包
   ```bash
   mkdir x86
   unzip x86_8.1.0.zip -d x86
   ```
   
   2. 进入目录
   ```bash 
   cd x86
   ```
   
   3. 安装whl包
   ```bash
   pip install msmonitor_plugin-*-cp39-*.whl
   ```
   
   4. 安装dynolog deb或rpm包
   ```
   rpm -ivh dynolog-*.rpm --nodeps
   # deb包则为 dpkg -i --force-overwrite dynolog*.deb
   ```

### 源码安装

### 1. clone 代码

```bash
git clone https://gitee.com/ascend/mstt.git
```

### 2. 安装依赖
dynolog的编译依赖，确保安装了以下依赖：
<table>
  <tr>
   <td>Language
   </td>
   <td>Toolchain
   </td>
  </tr>
  <tr>
   <td>C++
   </td>
   <td>gcc 8.5.0+
   </td>
  </tr>
  <tr>
   <td>Rust
   </td>
   <td>Rust >= 1.81
   </td>
  </tr>
</table>

- 安装rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

source $HOME/.cargo/env
```

- 安装ninja

```bash
# debian
sudo apt-get install -y cmake ninja-build

# centos
sudo yum install -y cmake ninja
```

- 安装protobuf (tensorboard_logger三方依赖，用于对接tensorboard展示)

```bash
# debian
sudo apt install -y protobuf-compiler libprotobuf-dev

# centos
sudo yum install -y protobuf protobuf-devel protobuf-compiler

# Python
pip install protobuf
```

- 安装openssl（RPC TLS认证）& 生成证书密钥
安装
```bash
# debian
sudo apt-get install -y openssl

# centos
sudo yum install -y openssl
```
dyno CLI与dynolog daemon之间的RPC通信使用TLS证书密钥加密，在启动dyno和dynolog二进制时需要指定证书密钥存放的路径，路径下需要满足如下结构和名称。
**用户应使用与自己需求相符的密钥生成和存储机制，并保证密钥安全性与机密性。**

服务端证书目录结构： 
```bash
server_certs
├── ca.crt (根证书，用于验证其他证书的合法性，必选)
├── server.crt (服务器端的证书，用于向客户端证明服务器身份，必选)
├── server.key (服务器端的私钥文件，与server.crt配对使用，支持加密，必选)
└── ca.crl (证书吊销列表，包含已被吊销的证书信息，可选)
```
客户端证书目录结构：
```bash
client_certs
├── ca.crt (根证书，用于验证其他证书的合法性，必选)
├── client.crt (客户端证书，用于向服务器证明客户端身份，必选)
├── client.key (客户端的私钥文件,与client.crt配对使用，支持加密，必选)
└── ca.crl (证书吊销列表，包含已被吊销的证书信息，可选)
```

### 3. 编译

- dynolog编译

默认编译生成dyno和dynolog二进制文件, -t参数可以支持将二进制文件打包成deb包或rpm包。

```bash
# 编译dyno和dynolog二进制文件
bash scripts/build.sh

# 编译deb包, 当前支持amd64和aarch64平台, 默认为amd64, 编译aarch64平台需要修改third_party/dynolog/scripts/debian/control文件中的Architecture改为arm64
bash scripts/build.sh -t deb

# 编译rpm包, 当前只支持amd64平台
bash scripts/build.sh -t rpm
```

- msmonitor-plugin wheel包编译

msmonitor-plugin wheel包提供IPCMonitor，MsptiMonitor等公共能力，使用nputrace和npu-monitor功能前必须安装该wheel包，具体编译安装指导可参考[msmonitor-plugin编包指导](./plugin/README.md)。

## 使用方式

- **说明**：
- 1. **NPU Monitor**功能和**Profiler trace dump** 功能不能同时开启。
- 2. **NPU Monitor**功能暂不支持图模式场景

### NPU Monitor功能
NPU Monitor基于MSPTI/MSTX能力开发，实现了轻量级在线监控能力，能够用于性能问题的初步定位。

**注意**：NPU Monitor功能开启时，不能同时开启Profiler trace dump功能。

```bash
dyno npu-monitor --help
```

- npu-monitor使用方式

```bash
dyno --certs-dir <CERT_DIR> npu-monitor [SUBCOMMANDS]
```

npu-monitor的SUBCOMMANDS（子命令）选项如下：

| 子命令                   | 参数类型 | 说明                                                                                                                                                  | PyTorch支持 | MindSpore支持 |    是否必选     |
|-----------------------|-------|------------------------------------------------------------------------------------------------------------------------------------------------------|:---------:|:-----------:|:-----------:|
| --npu-monitor-start   | action | 开启性能监控，设置参数后生效，默认不生效                                                                                                                                | Y | Y | N |
| --npu-monitor-stop    | action | 停止性能监控，设置参数后生效，默认不生效                                                                                                                                | Y | Y | N |
| --report-interval-s   | int | 性能监控数据上报周期，单位s，需要在启动时设置。默认值60                                                                                                                       | Y | Y | N |
| --mspti-activity-kind | String | 性能监控数据上报数据类型，可以设置单个或多个，多个类型以逗号分隔，每次设置时刷新全局上报类型。可选值范围[`Marker`, `Kernel`, `API`, `Hccl`, `Memory`, `MemSet`, `MemCpy`, `Communication`] , 默认值`Marker` | Y | Y | N |

- npu-monitor使用方法

Step 1： 拉起dynolog daemon进程
```bash
# 方法1和方法2 二选一
# 方法1：使用systemd拉起service
# 修改配置文件/etc/dynolog.gflags, 使能ipc_monitor
echo "--enable_ipc_monitor" | sudo tee -a /etc/dynolog.gflags
sudo systemctl start dynolog

# 方法2：命令行执行
dynolog --enable-ipc-monitor --certs-dir /home/server_certs

# 使用Tensorboard上报数据需要指定参数：--metric_log_dir, 指定Tensorboard文件落盘文件
# dynolog daemon的日志路径为：/var/log/dynolog.log
```

Step 2：在训练任务拉起窗口使能dynolog环境变量
```bash
export MSMONITOR_USE_DAEMON=1
```

Step 3: 配置Msmonitor日志路径(可选，默认路径为当前目录下的msmonitor_log)
```bash
export MSMONITOR_LOG_PATH=<LOG PATH>
# 示例：
export MSMONITOR_LOG_PATH=/tmp/msmonitor_log
```

Step 4: 拉起训练任务
```bash
# 训练任务拉起前需要设置LD_PRELOAD
# 示例：export LD_PRELOAD=/usr/local/Ascend/ascend-toolkit/latest/lib64/libmspti.so
export LD_PRELOAD=<CANN toolkit安装路径>/ascend-toolkit/latest/lib64/libmspti.so

# 训练任务中需要使用pytorch的优化器/继承原生优化器
bash train.sh
```

Step 5：使用dyno CLI使能npu-monitor
```bash
# 示例1：开启性能监控，使用默认配置
dyno --certs-dir /home/client_certs npu-monitor --npu-monitor-start

# 示例2：暂停性能监控
dyno --certs-dir /home/client_certs npu-monitor --npu-monitor-stop

# 示例3：性能监控过程中修改配置
# 上报周期30s, 上报数据类型Marker和Kernel
dyno --certs-dir /home/client_certs npu-monitor --report-interval-s 30 --mspti-activity-kind Marker,Kernel

# 示例4：性能监控开启时修改配置
# 上报周期30s, 上报数据类型Marker和Kernel
dyno --certs-dir /home/client_certs npu-monitor --npu-monitor-start --report-interval-s 30 --mspti-activity-kind Marker,Kernel

# 示例5：多机场景下性能监控开启时修改配置
# 多机场景下向特定机器x.x.x.x发送参数信息，参数表示上报周期30s, 上报数据类型Marker和Kernel
dyno --certs-dir /home/client_certs --hostname x.x.x.x npu-monitor --npu-monitor-start --report-interval-s 30 --mspti-activity-kind Marker,Kernel
```

Step6: 观测Tensorboard上报数据
```
# Tensorboard存储数据路径在指定参数metric_log_dir下
# 请确保安装了Tensorboard：

pip install tensorboard

# 然后运行：
# metric_log_dir为启动守护进程时所指定参数
tensorboard --logdir={metric_log_dir} 

# 打开浏览器访问http://localhost:6006即可看到对应可视化图表, 其中6006为tensorboard默认端口
```
> tensorboard 具体使用参数见https://github.com/tensorflow/tensorboard

### Profiler trace dump功能
Profiler trace dump功能基于dynolog开发，实现类似于动态profiling的动态触发Ascend Pytorch Profiler采集profiling的功能。用户基于dyno CLI命令行可以动态触发指定节点的训练进程trace dump。

- 查看dynolog支持的命令和帮助

```bash
dynolog --help
```

dynolog命令参数如下，更多dynolog原生参数请通过--help查看。

| 命令 | 参数类型   | 说明                                                   |    是否必选     |
|---|--------|------------------------------------------------------|:-----------:|
| --enable-ipc-monitor  | action | 是否启用IPC监控功能，用于与dyno进行通信，设置参数开启，默认不开启  |      N      |
| --port      |  i32   | dynolog daemon进程监听的端口号，默认值1778       |      N      |
| --certs-dir | String | 用于指定dyno与dynolog RPC通信时TLS证书的路径，当值为`NO_CERTS`时不使用证书校验|      Y      |


- 查看dyno支持的命令和帮助

```bash
dyno --help
```

dyno命令参数如下，更多dyno原生参数请通过--help查看。

| 命令          | 参数类型   | 说明                                   |    是否必选    |
|-------------|--------|--------------------------------------|:----------:|
| --hostname  | String | dynolog daemon所在主机的标识名称，默认值localhost |     N      |
| --port      |  i32   | dynolog daemon进程监听的端口号，默认值1778       |     N      |
| --certs-dir |  String   | 用于指定dyno与dynolog RPC通信时TLS证书的路径，当值为`NO_CERTS`时不使用证书校验|     Y      |

- 查看nputrace支持的命令和帮助

```bash
dyno nputrace --help
```

- nputrace使用方式

```bash
dyno --certs-dir <CERT_DIR> nputrace [SUBCOMMANDS] --log-file <LOG_FILE>
```

nputrace的SUBCOMMANDS（子命令）选项如下：

| 子命令                   | 参数类型 | 说明                                                                                                                                                                                                                                | PyTorch支持 | MindSpore支持 |  是否必选  |
|-----------------------|-------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------:|:-----------:|:----------:|
| --job-id              | u64 | 采集任务的job id，默认值0，dynolog原生参数                                                                                                                                                                                                      |     N     |      N      |  N |
| --pids                | String | 采集任务的pid列表，多个pid用逗号分隔，默认值0，dynolog原生参数                                                                                                                                                                                            |     N     |      N      |  N |
| --process-limit       | u64 | 最大采集进程的数量，默认值3，dynolog原生参数                                                                                                                                                                                                        |     N     |      N      | N |
| --profile-start-time  | u64 | 用于同步采集的Unix时间戳，单位毫秒，默认值0，dynolog原生参数                                                                                                                                                                                              |     N     |      N      | N |
| --duration-ms         | u64 | 采集的周期，单位毫秒，默认值500，dynolog原生参数                                                                                                                                                                                                     |     N     |      N      | N |
| --iterations          | i64 | 采集总迭代数，默认值-1，dynolog原生参数，需与start-step参数同时指定                                                                                                                                                                                       |     Y     |      Y      |  Y |
| --log-file            | String | 采集落盘的路径                                                                                                                                                                                                                           |     Y     |      Y      |  Y |
| --start-step          | u64 | 开始采集的迭代数，默认值0                                                                                                                                                                                                                     |     Y     |      Y      | Y  |
| --record-shapes       | action | 是否采集算子的InputShapes和InputTypes，设置参数采集，默认不采集                                                                                                                                                                                        |     Y     |      Y      |  N |
| --profile-memory      | action | 是否采集算子内存信息，设置参数采集，默认不采集                                                                                                                                                                                                           |     Y     |      Y      | N |
| --with-stack          | action | 是否采集Python调用栈，设置参数采集，默认不采集                                                                                                                                                                                                        |     Y     |      Y      | N |
| --with-flops          | action | 是否采集算子flops，设置参数采集，默认不采集                                                                                                                                                                                                          |     Y     |      N      | N |
| --with-modules        | action | 是否采集modules层级的Python调用栈，设置参数采集，默认不采集                                                                                                                                                                                              |     Y     |      N      | N |
| --analyse             | action | 采集后是否自动解析，设置参数解析，默认不解析                                                                                                                                                                                                            |     Y     |      Y      | N |
| --l2-cache            | action | 是否采集L2 Cache数据，设置参数采集，默认不采集                                                                                                                                                                                                       |     Y     |      Y      | N |
| --op-attr             | action | 是否采集算子属性信息，设置参数采集，默认不采集                                                                                                                                                                                                           |     Y     |      N      | N |
| --msprof-tx           | action | 是否使能MSTX，设置参数采集，默认不使能                                                                                                                                                                                                             |     Y     |      Y      | N |
| --mstx-domain-include | Option<String> | 使能--msprof-tx采集mstx打点数据的情况下，配置该开关，设置实际采集的domain范围，与--mstx-domain-exclude参数互斥，若同时设置，则只有--mstx-domain-include生效。该参数为可选参数，默认不使能。可配置一个或多个domain，例如：--mstx-domain-include domain1, domain2                                             |     Y     |      Y      |  N |
| --mstx-domain-exclude | Option<String> | 使能--msprof-tx采集mstx打点数据的情况下，配置该开关，设置实际不采集的domain范围，与--mstx-domain-include参数互斥，若同时设置，则只有--mstx-domain-include生效。该参数为可选参数，默认不使能。可配置一个或多个domain，例如：--mstx-domain-exclude domain1, domain2                                            |     Y     |      Y      | N |
| --data-simplification | String | 解析完成后是否数据精简，可选值范围[`true`, `false`]，默认值`true`                                                                                                                                                                                      |     Y     |      Y      | N |
| --activities          | String | 控制CPU、NPU事件采集范围，可以设置单个或多个，多个类型以逗号分隔，可选值范围[`CPU`, `NPU`]，默认值`CPU,NPU`                                                                                                                                                              |     Y     |      Y      | N |
| --profiler-level      | String | 控制profiler的采集等级，可选值范围[`Level_none`, `Level0`, `Level1`, `Level2`]，默认值`Level0`                                                                                                                                                     |     Y     |      Y      | N |
| --aic-metrics         | String | AI Core的性能指标采集项，可选值范围[`AiCoreNone`, `PipeUtilization`, `ArithmeticUtilization`, `Memory`, `MemoryL0`, `ResourceConflictRatio`, `MemoryUB`, `L2Cache`, `MemoryAccess`]，默认值`AiCoreNone`                                             |     Y     |      Y      | N |
| --export-type         | String | profiler解析导出数据的类型，可选值范围[`Text`, `Db`]，默认值`Text`                                                                                                                                                                                   |     Y     |      Y      | N |
| --gc-detect-threshold | Option<f32> | GC检测阈值，单位ms，只采集超过阈值的GC事件。该参数为可选参数，默认不设置时不开启GC检测                                                                                                                                                                                   |     Y     |      N      | N |
| --host-sys            | String | 采集[host侧系统数据](https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/T&ITools/Profiling/atlasprofiling_16_0014.html)(CPU利用率、内存利用率、磁盘I/O利用率、网络I/O利用率等)。该参数为可选参数，可选值范围[`cpu`, `mem`, `disk`, `network`, `osrt`] , 默认不设置时不开启host侧系统数据采集 |     Y     |      Y      | N |
| --sys-io              | action | 采集NIC、ROCE数据。该参数为可选参数，设置参数采集，默认不采集                                                                                                                                                                                                |     Y     |      Y      | N |
| --sys-interconnection | action | 采集集合通信带宽数据（HCCS）、PCIe、片间传输带宽数据。该参数为可选参数，设置参数采集，默认不采集                                                                                                                                                                              |     Y     |      Y      | N |

- nputrace使用方法

Step 0: 参考[3.编译](./README.md#3-编译)章节完成dynolog的编译，以及dynolog_npu_plugin wheel包的编译和安装。

Step 1：拉起dynolog daemon进程
```bash
# 方法1和方法2 二选一
# 方法1：使用systemd拉起service
# 修改配置文件/etc/dynolog.gflags, 使能ipc_monitor
echo "--enable_ipc_monitor" | sudo tee -a /etc/dynolog.gflags
sudo systemctl start dynolog

# 方法2：命令行执行
dynolog --enable-ipc-monitor --certs-dir /home/server_certs

#dynolog daemon的日志路径为：/var/log/dynolog.log
```

Step 2：在训练任务拉起窗口使能dynolog环境变量
```bash
export MSMONITOR_USE_DAEMON=1
```

Step 3: 拉起训练任务
```bash
# 训练任务中需要使用pytorch的优化器/继承原生优化器
bash train.sh
```

Step 4：使用dyno CLI动态触发trace dump
```bash
# 示例1：从第10个step开始采集，采集2个step，采集框架、CANN和device数据，同时采集完后自动解析以及解析完成不做数据精简，落盘路径为/tmp/profile_data
dyno --certs-dir /home/client_certs nputrace --start-step 10 --iterations 2 --activities CPU,NPU --analyse --data-simplification false --log-file /tmp/profile_data

# 示例2：从第10个step开始采集，采集2个step，只采集CANN和device数据，同时采集完后自动解析以及解析完成后开启数据精简，落盘路径为/tmp/profile_data
dyno --certs-dir /home/client_certs nputrace --start-step 10 --iterations 2 --activities NPU --analyse --data-simplification true --log-file /tmp/profile_data

# 示例3：从第10个step开始采集，采集2个step，只采集CANN和device数据，只采集不解析，落盘路径为/tmp/profile_data
dyno --certs-dir /home/client_certs nputrace --start-step 10 --iterations 2 --activities NPU --log-file /tmp/profile_data

# 示例4：多机场景下向特定机器x.x.x.x发送参数信息，参数表示从第10个step开始采集，采集2个step，只采集CANN和device数据，只采集不解析，落盘路径为/tmp/profile_data
dyno --certs-dir /home/client_certs --hostname x.x.x.x nputrace --start-step 10 --iterations 2 --activities NPU --log-file /tmp/profile_data
```


## [Mindspore框架下msMonitor的使用方法](./docs/mindspore_adapter.md)

## 附录

[安全声明](./docs/security_statement.md)