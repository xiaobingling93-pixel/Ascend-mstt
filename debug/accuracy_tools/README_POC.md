# 精度工具

本手册主要介绍精度预检工具和ptdbg_ascend精度工具合一软件包的安装和工具命令行使用指导。

## 工具安装

精度工具合一软件包名称为：`ascend_training_accuracy_tools-{version}-py3-none-any.whl`

1. whl包获取。

   请通过下表链接下载ptdbg_ascend精度工具whl包。

   | ptdbg_ascend版本 | 发布日期   | 支持PyTorch版本 | 下载链接                                                     | 校验码                                                       |
   | ---------------- | ---------- | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
   | 0.0.1            | 2024-03-08 | 1.11.0/2.0/2.1  | [ascend_training_accuracy_tools-0.0.1-py3-none-any.whl](https://ptdbg.obs.myhuaweicloud.com/att/0.0/ascend_training_accuracy_tools-0.0.1-py3-none-any.whl) | 7d5978720f00772546f324a577e842ca66f42342ea5d99897d5407e5c4e71b4a |

2. whl包校验。

   1. 根据以上下载链接下载whl包到Linux安装环境。

   2. 进入whl包所在目录，执行如下命令。

      ```bash
      sha256sum {name}.whl
      ```

      {name}为whl包名称。

      若回显呈现对应版本whl包一致的**校验码**，则表示下载了正确的ptdbg_ascend精度工具whl安装包。示例如下：

      ```bash
      sha256sum ascend_training_accuracy_tools-0.0.1-py3-none-any.whl
      7d5978720f00772546f324a577e842ca66f42342ea5d99897d5407e5c4e71b4a *ascend_training_accuracy_tools-0.0.1-py3-none-any.whl
      ```

3. 执行如下命令进行安装。

   ```bash
   pip3 install ./ascend_training_accuracy_tools-{version}-py3-none-any.whl
   ```

   若为覆盖安装，请在命令行末尾增加“--force-reinstall”参数强制安装，例如：

   ```bash
   pip3 install ./ascend_training_accuracy_tools-{version}-py3-none-any.whl --force-reinstall
   ```

   提示如下信息则表示安装成功。

   ```bash
   Successfully installed ascend_training_accuracy_tools-{version}
   ```


## 工具使用

安装精度工具合一软件包后，精度工具支持使用命令行启动各种功能（除ptdbg_ascend工具的dump和精度比对操作）。命令格式如下：

```bash
atat [-h] parse run_ut multi_run_ut benchmark_compare run_overflow_check
```

| 参数               | 说明                                                         |
| ------------------ | ------------------------------------------------------------ |
| parse              | ptdbg_ascend.parse数据解析功能入口，执行atat parse命令后进入parse交互式界面，更多参数请参见《[ptdbg_ascend精度工具功能说明](https://gitee.com/ascend/att/tree/master/debug/accuracy_tools/ptdbg_ascend/doc)》的“ptdbg_ascend.parse数据解析功能”。 |
| run_ut             | 预检工具run_ut功能，可以通过atat run_ut命令执行精度预检操作，更多参数请参见《[Ascend模型精度预检工具](https://gitee.com/ascend/att/tree/master/debug/accuracy_tools/api_accuracy_checker)》的“执行预检”。 |
| multi_run_ut       | 预检工具multi_run_ut功能，可以通过atat multi_run_ut命令执行多线程预检操作，更多参数请参见《[Ascend模型精度预检工具](https://gitee.com/ascend/att/tree/master/debug/accuracy_tools/api_accuracy_checker)》的“multi_run_ut多线程预检”。 |
| benchmark_compare  | 预检工具预检结果比对功能，可以通过atat benchmark_compare命令执行预检结果比对操作，更多参数请参见《[Ascend模型精度预检工具](https://gitee.com/ascend/att/tree/master/debug/accuracy_tools/api_accuracy_checker)》的“multi_run_ut多线程预检”。 |
| run_overflow_check | 溢出解析工具，可以通过atat run_overflow_check命令执行溢出API解析操作，更多参数请参见《[Ascend模型精度预检工具](https://gitee.com/ascend/att/tree/master/debug/accuracy_tools/api_accuracy_checker)》的“溢出解析工具”。 |