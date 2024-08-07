# MindStudio精度调试工具

MindStudio精度调试工具（MindStudio Probe），简称msprobe，是MindStudio Training Tools工具链下精度调试部分的工具包。主要包括精度预检和精度比对等子工具，当前适配场景包括PyTorch和MindSpore。

## 工具安装

精度工具合一软件包名称：`mindstudio_probe-{version}-py3-none-any.whl`

### pip安装
   ```shell
   pip install mindstudio-probe
   ```
使用`pip install mindstudio-probe==版本号`可安装指定版本的包。

pip命令会自动安装最新的包及其配套依赖。

提示如下信息则表示安装成功。

```bash
Successfully installed mindstudio_probe-{version}
```

### 下载whl包安装
1. 使用pip命令安装依赖：

   1. 根据实际环境安装torch或mindspore

   2. 安装numpy、openpyxl、pandas、PyYAML、rich、tqdm、einops、matplotlib


   若环境中已安装部分依赖，不需要重复安装。

2. whl包获取。

   请通过下表链接下载工具whl包。

   | 版本  | 发布日期   | 支持PyTorch版本 | 下载链接                                                     | 校验码                                                       |
   | ----- | ---------- | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
   | 1.0.1 | 2024-07-25 | 2.0/2.1/2.2     | [mindstudio_probe-1.0.1-py3-none-any.whl](https://ptdbg.obs.myhuaweicloud.com/msprobe/1.0/mindstudio_probe-1.0.1-py3-none-any.whl) | b699e224e4d4e3bcf9412c54fa858a1ee370f0d7a2bc69cb3f1273ac14a6dc82 |
   | 1.0   | 2024-07-09 | 2.0/2.1/2.2     | [ascend_training_accuracy_tools-1.0-py3-none-any.whl](https://ptdbg.obs.myhuaweicloud.com/att/1.0/ascend_training_accuracy_tools-1.0-py3-none-any.whl) | 5016dfe886c5d340ec6f60a959673355855f313c91f100680da814efb49f8e81 |
   | 0.0.3 | 2024-06-11 | 2.0/2.1/2.2     | [ascend_training_accuracy_tools-0.0.3-py3-none-any.whl](https://ptdbg.obs.myhuaweicloud.com/att/0.0/ascend_training_accuracy_tools-0.0.3-py3-none-any.whl) | f46d9714704859e2d67861a65bbb3c76b0a250cf6e238b978b5b959ab1fe125a |
   | 0.0.2 | 2024-05-23 | 2.0/2.1/2.2     | [ascend_training_accuracy_tools-0.0.2-py3-none-any.whl](https://ptdbg.obs.myhuaweicloud.com/att/0.0/ascend_training_accuracy_tools-0.0.2-py3-none-any.whl) | 2e35809bde559e9c4d2f16a02ccde779ed9e436bb65fded0b7ebaf6ac2c88d93 |
   | 0.0.1 | 2024-03-15 | 2.0/2.1         | [ascend_training_accuracy_tools-0.0.1-py3-none-any.whl](https://ptdbg.obs.myhuaweicloud.com/att/0.0/ascend_training_accuracy_tools-0.0.1-py3-none-any.whl) | 5801510d4e827e4859bc9a5aca021e4d30c2ea42d60a4c8ad0c2baab1b7782c9 |

3. whl包校验。

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

4. 执行如下命令进行安装。

   ```bash
   pip3 install ./mindstudio_probe-{version}-py3-none-any.whl
   ```

   若为覆盖安装，请在命令行末尾增加“--force-reinstall”参数强制安装，例如：

   ```bash
   pip3 install ./mindstudio_probe-{version}-py3-none-any.whl --force-reinstall
   ```

   提示如下信息则表示安装成功。

   ```bash
   Successfully installed mindstudio_probe-{version}
   ```

### 从源码安装
1. 克隆或者下载项目源代码
   
   ```shell
   git clone https://gitee.com/ascend/mstt.git
   cd debug/accuracy_tools
   ```
   
2. 安装setuptools和wheel
   
   ```shell
   pip install setuptools wheel
   ```
   
3. 安装msprobe
   
   ```shell
   python setup.py install
   ```
   提示出现如下信息则表示源码安装成功。
   ```shell
   Finished processing dependencies for mindstudio-probe=={version}
   ```

### 查看msprobe工具信息

执行如下命令查看msprobe工具信息。

```bash
pip show mindstudio-probe
```

输出结果如下示例：

```bash
Name: mindstudio-probe
Version: 1.0
Summary: This is a pytorch precision comparison tools
Home-page:
Author:
Author-email:
License:
Location: /home/xx/anaconda3/envs/pt21py38/lib/python3.8/site-packages
Requires: numpy, openpyxl, pandas, pyyaml, rich, tqdm, wheel
Required-by:
```

关键字段含义：

- Name：工具名称。
- Version：工具版本号。
- Summary：工具概述。
- Location：工具安装路径。
- Requires：工具依赖。

## 工具使用

安装msprobe工具后，可以按照如下思路选择合适的子工具进行精度调试：

1. 判断框架场景。

   当前支持PyTorch和MindSpore场景。

2. 执行数据采集。 

   工具通过在训练脚本中添加PrecisionDebugger接口的方式对API执行精度数据dump操作。

   PyTorch场景：详见[PyTorch_精度数据采集](./pytorch/doc/dump.md)。

   MindSpore场景：详见[MindSpore_精度数据采集](./mindspore/doc/dump.md)。

3. 执行精度预检。

   在昇腾NPU上扫描用户训练模型中所有API，进行API复现，给出精度情况的诊断和分析。

   PyTorch场景：详见[PyTorch_精度预检工具](./pytorch/doc/api_accuracy_checker.md)。

   MindSpore场景：暂不支持。

4. 执行精度比对。

   进行PyTorch整网API粒度的数据dump、精度比对和溢出检测，从而定位训练场景下的精度问题。

   PyTorch场景：详见[PyTorch_精度比对工具](./pytorch/doc/ptdbg_ascend_overview.md)。

   MindSpore场景：暂不支持。

5. 执行溢出解析。

   溢出解析是在执行精度数据dump时，配置了溢出检测dump，那么对于输入正常但输出存在溢出的API，可以判断是否为正常溢出。

   PyTorch场景：详见[PyTorch_溢出解析工具](./pytorch/doc/run_overflow_check.md)。

   MindSpore场景：暂不支持。

6. 执行数据解析。

   用于比对前后两次NPU ACL层级dump数据的一致性。

   PyTorch场景：详见[PyTorch_数据解析工具](./pytorch/doc/parse_tool.md)。

   MindSpore场景：暂不支持。

上述流程中的工具均为msprobe工具的子工具，使用相同的命令行，格式如下：

精度预检工具

```bash
msprobe -f <framework> run_ut [-h]
```

```bash
msprobe -f <framework> multi_run_ut [-h]
```

```bash
msprobe -f <framework> api_precision_compare [-h]
```

精度比对工具

```bash
msprobe -f <framework> compare [-h]
```

溢出解析工具

```bash
msprobe -f <framework> run_overflow_check [-h]
```

数据解析工具

```bash
msprobe -f <framework> parse [-h]
```

| 参数 | 说明                                                   |
| ---- | ------------------------------------------------------ |
| -f   | 框架，请按所使用框架配置，当前支持pytorch或mindspore。 |
| -h   | 帮助信息。                                             |

## 贡献

push代码前，请务必保证已经完成了基础功能测试和网络测试。

## Release Notes

Release Notes请参见[RELEASE](RELEASE.md)。