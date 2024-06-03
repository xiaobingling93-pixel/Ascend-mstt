# MindStudio精度调试工具

MindStudio精度调试工具（ascend_training_accuracy_tools），简称atat，是ATT工具链下精度调试部分的工具包。主要包括精度预检和精度比对等子工具，当前适配场景包括PyTorch和MindSpore。

## 工具安装

精度工具合一软件包名称：`ascend_training_accuracy_tools-{version}-py3-none-any.whl`

1. whl包获取。

   请通过下表链接下载工具whl包。

   | 版本  | 发布日期   | 支持PyTorch版本    | 下载链接                                                     | 校验码                                                       |
   | ----- | ---------- | ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
   | 0.0.2 | 2024-05-23 | 1.11.0/2.0/2.1/2.2 | [ascend_training_accuracy_tools-0.0.2-py3-none-any.whl](https://ptdbg.obs.myhuaweicloud.com/att/0.0/ascend_training_accuracy_tools-0.0.2-py3-none-any.whl) | 2e35809bde559e9c4d2f16a02ccde779ed9e436bb65fded0b7ebaf6ac2c88d93 |
   | 0.0.1 | 2024-03-15 | 1.11.0/2.0/2.1     | [ascend_training_accuracy_tools-0.0.1-py3-none-any.whl](https://ptdbg.obs.myhuaweicloud.com/att/0.0/ascend_training_accuracy_tools-0.0.1-py3-none-any.whl) | 5801510d4e827e4859bc9a5aca021e4d30c2ea42d60a4c8ad0c2baab1b7782c9 |

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
      5801510d4e827e4859bc9a5aca021e4d30c2ea42d60a4c8ad0c2baab1b7782c9 *ascend_training_accuracy_tools-0.0.1-py3-none-any.whl
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

安装atat工具后，可以按照如下思路选择合适的子工具进行精度调试：

1. 判断框架场景。

   当前支持PyTorch和MindSpore场景。

2. 执行数据采集。 

   工具通过在训练脚本中添加PrecisionDebugger接口的方式对API执行精度数据dump操作。

   PyTorch场景：详见[PyTorch_精度数据采集](./Pytorch/doc/dump.md)。

   MindSpore场景：详见[MindSpore_精度数据采集](./MindSpore/doc/dump.md)。

3. 执行精度预检。

   在昇腾NPU上扫描用户训练模型中所有API，进行API复现，给出精度情况的诊断和分析。

   PyTorch场景：详见[PyTorch_精度预检工具](./Pytorch/doc/api_accuracy_checker.md)。

   MindSpore场景：暂不支持。

4. 执行精度比对。

   进行PyTorch整网API粒度的数据dump、精度比对和溢出检测，从而定位训练场景下的精度问题。

   PyTorch场景：详见[PyTorch_精度比对工具](./Pytorch/doc/ptdbg_ascend_overview.md)。

   MindSpore场景：暂不支持。

5. 执行溢出解析。

   溢出解析是在执行精度数据dump时，配置了溢出检测dump，那么对于输入正常但输出存在溢出的API，可以判断是否为正常溢出。

   PyTorch场景：详见[PyTorch_溢出解析工具](./Pytorch/doc/run_overflow_check.md)。（暂不支持）

   MindSpore场景：暂不支持。

6. 执行数据解析。

   用于比对前后两次NPU ACL层级dump数据的一致性。

   PyTorch场景：详见[PyTorch_数据解析工具](./Pytorch/doc/parse_tool.md)。

   MindSpore场景：暂不支持。

上述流程中的工具均为atat工具的子工具，使用相同的命令行，格式如下：

```bash
atat [-h] -f <framework> parse run_ut multi_run_ut api_precision_compare run_overflow_check
```

| 参数 | 说明                                     |
| ---- | ---------------------------------------- |
| -f   | 框架，当前支持配置为pytorch和mindspore。 |
| -h   | 帮助信息。                               |

其他参数在上述对应的工具手册中详细介绍。

## 贡献

push代码前，请务必保证已经完成了基础功能测试和网络测试。

## Release Notes

Release Notes请参见[RELEASE](RELEASE.md)。