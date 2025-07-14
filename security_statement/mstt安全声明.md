# mstt安全声明

## 系统安全加固

建议用户在系统中配置开启ASLR（级别2 ），又称**全随机地址空间布局随机化**，可参考以下方式进行配置：

    echo 2 > /proc/sys/kernel/randomize_va_space

## 运行用户建议

出于安全性及权限最小化角度考虑，不建议使用root等管理员类型账户使用精度工具、性能工具。

## 文件权限控制

 **1. ** msMonitor在线监控、msprobe在线预检功能在默认安全模式下，需提供根证书、服务端证书、服务端私钥文件、证书吊销列表文件所在的目录，用户需保证目录权限为700、证书等文件权限为600。

 **2. ** 用户向工具提供输入文件输入时，建议提供的文件属主与工具进程属主一致，且文件权限他人不可修改（包括group、others）。工具落盘文件权限默认他人不可写，用户可根据需要自行对生成后的相关文件进行权限控制。

 **3. ** 用户安装和使用过程需要做好权限控制，建议参考文件权限参考进行设置。


##### 文件权限参考

| 类型                               | Linux权限参考最大值 |
| ---------------------------------- | ------------------- |
| 用户主目录                         | 750（rwxr-x---）    |
| 程序文件(含脚本文件、库文件等)     | 550（r-xr-x---）    |
| 程序文件目录                       | 550（r-xr-x---）    |
| 配置文件                           | 640（rw-r-----）    |
| 配置文件目录                       | 750（rwxr-x---）    |
| 日志文件(记录完毕或者已经归档)     | 440（r--r-----）    |
| 日志文件(正在记录)                 | 640（rw-r-----）    |
| 日志文件目录                       | 750（rwxr-x---）    |
| Debug文件                          | 640（rw-r-----）    |
| Debug文件目录                      | 750（rwxr-x---）    |
| 临时文件目录                       | 750（rwxr-x---）    |
| 维护升级文件目录                   | 770（rwxrwx---）    |
| 业务数据文件                       | 640（rw-r-----）    |
| 业务数据文件目录                   | 750（rwxr-x---）    |
| 密钥组件、私钥、证书、密文文件目录 | 700（rwx------）    |
| 密钥组件、私钥、证书、加密密文     | 600（rw-------）    |
| 加解密接口、加解密脚本             | 500（r-x------）    |

## 数据安全声明

​    工具使用过程中需要加载和保存数据，部分接口直接或间接使用风险模块pickle，可能存在数据风险，如torch.load等接口，可参考[torch.load](https://pytorch.org/docs/main/generated/torch.load.html#torch.load)了解具体风险。

## 构建安全声明

​    msmonitor、msprobe支持源码编译安装，在编译时会下载依赖第三方库并执行构建shell脚本，在编译过程中会产生临时程序文件和编译目录。用户可根据需要自行对源代码目录内的文件进行权限管控降低安全风险，用户在构建过程中可根据需要修改构建脚本以避免相关安全风险，并注意构建结果的安全。

## 运行安全声明

1. 工具加载数据集时，如数据集加载内存大小超出内存容量限制，可能引发错误并导致进程意外退出；采集时间过长导致生成数据超过磁盘空间大小时，可能会导致异常退出。

2. 工具在运行异常时会退出进程并打印报错信息，属于正常现象。建议用户根据报错提示定位具体错误原因，包括查看日志文件，采集解析过程中生成的结果文件等方式。

3. 精度工具msprobe：

   使用前提：被采集对象的python源码应可读可执行以便获取公开调用栈等公开信息

   使用场景：用户需要进行模型精度分析时，可以在模型训练开脚本中添加msprobe的dump接口，执行训练的同时采集精度数据，完成训练后直接输出精度数据文件。数据文件内容包含模型中的API数据、模型结构以及API调用的堆栈信息（方便定位到在精度问题的API时，能快速找到该API在模型中的位置）。

   风险提示：使用该功能会在本地生成精度数据，用户需加强对相关精度数据文件的保护，请在需要模型精度分析时使用，分析完毕后及时关闭。

## 公网地址声明

在mstt仓工具的配置文件和脚本中存在的[公网地址](#公网地址)

##### 公网地址
| 类型     | 开源代码地址 | 文件名                                                       | 公网IP地址/公网URL地址/域名/邮箱地址                         | 用途说明                                   |
| -------- | ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------ |
| 开源软件 | -            | 所有文件                                                      | http://www.apache.org/licenses/LICENSE-2.0                 | 文件头中的license信息说明  |
| 开源软件 | -            | debug/accuracy_tools/setup.py                                | pmail_mindstudio@huawei.com                                  | 用于软件包的author-email信息               |
| 开源软件 | -            | debug/accuracy_tools/setup.py                                | https://gitee.com/ascend/mstt/tree/master/debug/accuracy_tools/msprobe | 用于软件包的home-page信息                  |
| 开源软件 | -            | debug/accuracy_tools/cmake/config.ini                        | https://gitee.com/mirrors/googletest/repository/archive/release-1.12.1.tar.gz | 开源软件下载                               |
| 开源软件 | -            | debug/accuracy_tools/cmake/config.ini                        | https://gitee.com/sinojelly/mockcpp/repository/archive/v2.7.zip | 开源软件下载                               |
| 开源软件 | -            | debug/accuracy_tools/cmake/config.ini                        | https://gitee.com/mirrors/JSON-for-Modern-CPP/repository/archive/v3.10.1.zip | 开源软件下载                               |
| 开源软件 | -            | debug/accuracy_tools/cmake/config.ini                        | https://gitee.com/mirrors/openssl/repository/archive/OpenSSL_1_1_1k.tar.gz | 开源软件下载                               |
| 开源软件 | -            | debug/accuracy_tools/cmake/config.ini                        | https://gitee.com/mirrors/protobuf_source/repository/archive/v3.15.0.tar.gz | 开源软件下载                               |
| 开源软件 | -            | /.gitmodules                                                 | [https://github.com/facebookincubator/dynolog.git](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Ffacebookincubator%2Fdynolog.git) | 在线监控底座                               |
| 开源软件 | -            | /msmonitor/dynolog_npu/cmake/config.ini                      | https://gitee.com/mirrors/openssl.git                        | 开源软件下载                               |
| 开源软件 | -            | /msmonitor/scripts/build.sh                                  | [https://github.com/RustingSword/tensorboard_logger.git](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2FRustingSword%2Ftensorboard_logger.git) | 开源软件下载                               |
| 开源软件 | -            | /msmonitor/README.md                                         | [https://github.com/tensorflow/tensorboard](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Ftensorflow%2Ftensorboard) | tensorboard官网教程                        |
| 开源软件 | -            | /profiler/msprof_analyze/advisor/config/config.ini           | [https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/devaids/auxiliarydevtool/atlasprofiling_16_0038.html](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2Fcanncommercial%2F80RC2%2Fdevaids%2Fauxiliarydevtool%2Fatlasprofiling_16_0038.html) | MindStudio Ascend PyTorch Profiler参考示例 |
| 开源软件 | -            | /profiler/msprof_analyze/advisor/config/config.ini           | https://gitee.com/ascend/mstt/blob/master/profiler/msprof_analyze/advisor/doc/Samples of Fused Operator API Replacement.md" | Advisor优化手段参考示例                    |
| 开源软件 | -            | /profiler/msprof_analyze/advisor/config/config.ini           | [https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/devaids/auxiliarydevtool/aoe_16_043.html](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2Fcanncommercial%2F80RC2%2Fdevaids%2Fauxiliarydevtool%2Faoe_16_043.html) | Advisor优化手段参考示例                    |
| 开源软件 | -            | /profiler/msprof_analyze/advisor/config/config.ini           | [https://www.mindspore.cn/lite/docs/en/master/use/cloud_infer/converter_tool_ascend.html#aoe-auto-tuning](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Flite%2Fdocs%2Fen%2Fmaster%2Fuse%2Fcloud_infer%2Fconverter_tool_ascend.html%23aoe-auto-tuning) | Advisor优化手段参考示例                    |
| 开源软件 | -            | /profiler/msprof_analyze/advisor/config/config.ini           | [https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptmigr/AImpug_000060.html](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2Fcanncommercial%2F700%2Fmodeldevpt%2Fptmigr%2FAImpug_000060.html) | Advisor优化手段参考示例                    |
| 开源软件 | -            | /profiler/msprof_analyze/config/config.ini                   | https://gitee.com/ascend/mstt/tree/master/profiler/msprof_analyze | msprof-analyze工具地址                     |
| 开源软件 | -            | /profiler/msprof_analyze/LICENSE                             | [http://www.apache.org/licenses/LICENSE-2.0](https://gitee.com/link?target=http%3A%2F%2Fwww.apache.org%2Flicenses%2FLICENSE-2.0) | 开源软件协议地址                           |
| 开源软件 | -            | /profiler/msprof_analyze/advisor/rules/aicpu_rules.yaml      | https://gitee.com/ascend/mstt/blob/master/profiler/msprof_analyze/advisor/doc/Samples of AI CPU Operator Replacement.md | AI CPU 算子替换样例                        |
| 开源软件 | -            | /profiler/msprof_analyze/advisor/rules/environment_variable_info.yaml | [https://support.huawei.com/enterprise/zh/doc/EDOC1100371278/5eeeed85?idPath=23710424](https://gitee.com/link?target=https%3A%2F%2Fsupport.huawei.com%2Fenterprise%2Fzh%2Fdoc%2FEDOC1100371278%2F5eeeed85%3FidPath%3D23710424) | 组网指南                                   |
| 开源软件 | -            | /profiler/msprof_analyze/config/config.ini                   | [pmail_mindstudio@huawei.com](https://gitee.com/link?target=mailto%3Apmail_mindstudio%40huawei.com) | 公网邮箱                                   |
| 开源软件 | -            | plugins/tensorboard-plugins/tb_graph_ascend/setup.py         | pmail_mindstudio@huawei.com                                                                         | MindStudio官方邮箱                         |
| 开源软件 | -            | plugins/tensorboard-plugins/tb_graph_ascend/setup.py         | https://gitee.com/ascend/mstt/tree/master/plugins/tensorboard-plugins/tb_graph_ascend               | 代码仓地址                                  |
| 开源软件 | -            | 非源码引入，只是在构建产物中包含                                | https://npms.io                                                                                     | 构建引入，注释                                   |
| 开源软件 | -            | 非源码引入，只是在构建产物中包含                                | https://github.com/webcomponents/shadycss/issues/193 | 构建引入，注释                                   |
| 开源软件 | -            | 非源码引入，只是在构建产物中包含                                | http://jsbin.com/temexa/4  | 构建引入，注释                                |
| 开源软件 | -            | 非源码引入，只是在构建产物中包含                                | https://fonts.googleapis.com/ | 构建引入，组件样式文件                                  |
| 开源软件 | -            | 非源码引入，只是在构建产物中包含                                | https://developer.mozilla.org/ | 构建引入，注释                                   |
| 开源软件 | -            | 非源码引入，只是在构建产物中包含                                | https://github.com/vaadin/vaadin-time-picker/issues/145 | 构建引入，注释                                   |
| 开源软件 | -            | 非源码引入，只是在构建产物中包含                                | http://codepen.io/shyndman/pen/ | 构建引入，注释                                   |

## 公开接口声明

mstt项目采用C++和Python联合开发，提供的对外接口均已在资料中公开，正式接口只提供Python接口，动态库不直接提供服务，暴露的接口为内部使用，不建议用户使用。

## 通信安全加固

 ** 1.在线监控、在线预检默认安全加密通信** 

msmonitor在线监控、msprobe在线预检功能在默认安全模式下，需提供根证书、服务端证书、服务端私钥文件、客户端证书、客户端私钥文件、证书吊销列表文件，用户需保证提供的证书文件的有效性、合法性，同时用户需保证证书所在的目录文件权限为700、证书等文件权限为600.

 ** 2.dynolog原生全零监听安全风险** 

 **背景：** msmonitor引入了开源第三方库dynolog。msmonitor对dynolog进行了NPU适配，适配引入的文件msmonitor/dynolog_npu/dynolog/src/rpc/SimpleJsonServer.cpp中包含全零监听代码，为保证工具功能和易用性，未对原生dynolog全零监听还没进行修改。

 **风险：** 该库的dynolog/src/rpc/SimpleJsonServer.cpp文件包含全零监听功能（bind to in6addr_any）,存在网络暴露安全风险。该安全风来源于dynolog开源三方库。

 ** 3.模型分级可视化tensorboard插件** 

 **背景：** 模型分级可视化插件（plugins/tensorboard-plugins/tb_graph_ascend）基于tensorboard底座开发和调试，安装插件完毕，需要启动tensorboard后使用。

 **风险：** tensorboard启动时可以通过--host指定IP，或通过--bind-all指定绑定在全零监听。由于分级可视化插件仅为tensorboard的插件，无法针对tensorboard自身服务进行安全加固，建议用户使用时确保环境安全。

 **风险消减措施：** 建议使用时绑定--host=127.0.0.1或localhost，并尽量避免使用root用户启动。如因远程访问场景需要将tensorboard服务启动在非localhost，建议用户确保环境自身的安全以及通过网络安全加固方案进行防护，如通过iptables等访问控制策略限制使用的客户端，或通过nginx等反向代理工具进行https加固。

## 通信矩阵

### 通信矩阵信息

| 序号 | 代码仓              | 功能                      | 源设备                        | 源IP                            | 源端口 | 目的设备                | 目的IP                        | 目的端口<br/>（侦听） | 协议 | 端口说明            | 端口配置 | 侦听端口是否可更改 | 认证方式 | 加密方式 | 所属平面 | 版本     | 特殊场景 | 备注 |
| :--- | :------------------ | :------------------------ | :---------------------------- | :------------------------------ | :----- | :---------------------- | :---------------------------- | :-------------------- | :--- | :------------------ | :------- | :----------------- | :------- | :------- | :------- | :------- | :------- | :--- |
| 1    | msMonitor           | dyno和dynolog RPC通信     | dyno客户端                    | 运行dyno客户端进程的服务器的ip  |        | dynolog服务端所在服务器 | dynolog服务端所在服务器的ip   | 1778                  | TCP  | RPC通信             | 不涉及   | 可修改             | 证书密钥 | TLS      | 业务面   | 所有版本 | 无       |      |
| 2    | tensorboard-plugins | TensorBoard底座前后端通信 | 访问TensorBoard浏览器所在机器 | 访问TensorBoard浏览器所在机器ip |        | TensorBoard服务所在机器 | TensorBoard服务所在服务器的ip | 6006                  | HTTP | tensorboard服务通信 | --port   | 可修改             |          |          | 业务面   | 所有版本 | 无       |      |