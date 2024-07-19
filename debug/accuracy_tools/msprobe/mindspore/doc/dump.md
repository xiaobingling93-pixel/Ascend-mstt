# **精度数据采集**

msprobe工具主要通过在训练脚本内添加dump接口并启动训练的方式来采集精度数据。

执行dump操作需要安装msprobe工具。详见《[MindStudio精度调试工具](../../README.md)》的“工具安装”章节。

## dump接口介绍

### PrecisionDebugger

**功能说明**

通过加载dump配置文件的方式来确定dump操作的详细配置。

可以在from msprobe.mindspore import PrecisionDebugger和模型初始化之间的任意位置添加该接口。

**原型**

```Python
PrecisionDebugger(config_path=None)
```

**参数说明**

| 参数名      | 说明                                                         | 是否必选 |
| ----------- | ------------------------------------------------------------ | -------- |
| config_path | 指定dump配置文件路径，String类型。参数示例："./config.json"。未配置该路径时，默认使用[config.json](../../config)文件的默认配置。config.json文件可以配置更多参数，若需要进行更多场景的精度数据dump，建议配置[config.json](../../config/config.json)文件。 | 否       |

### start函数

**功能说明**

启动函数。

**原型**

```Python
debugger.start()
```

该函数为类函数，可以使用debugger.start()也可以使用PrecisionDebugger.start()。

## 示例代码

```Python
from msprobe.mindspore import PrecisionDebugger
debugger = PrecisionDebugger(config_path="./config.json")
# 请勿将以上初始化流程插入到循环代码中
# 下面代码也可以用PrecisionDebugger.start()
debugger.start()
...
```

## dump结果文件介绍

训练结束后，工具将dump的数据保存在dump_path参数指定的目录下。

- level为L1时

  dump结果目录请参见MindSpore官网中的《[同步Dump数据对象目录](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.0rc2/debug/dump.html#%E5%90%8C%E6%AD%A5dump%E6%95%B0%E6%8D%AE%E5%AF%B9%E8%B1%A1%E7%9B%AE%E5%BD%95)》。

- level为L2时

  dump结果目录请参见MindSpore官网中的《[异步Dump数据对象目录](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.0rc2/debug/dump.html#%E5%BC%82%E6%AD%A5dump%E6%95%B0%E6%8D%AE%E5%AF%B9%E8%B1%A1%E7%9B%AE%E5%BD%95)》。

