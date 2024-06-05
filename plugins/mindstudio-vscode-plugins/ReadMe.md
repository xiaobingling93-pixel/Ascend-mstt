# MindStudio Operator Debug VSCode Plugin

### 介绍
MindStudio Operator Debug VSCode Plugin插件基于MindStudio Debugger(msdebug调试器)提供的底层调试昇腾算子能力，支持远程调试C/C++与昇腾算子程序。
MindStudio Operator Debug VSCode Plugin插件[源码仓](https://gitee.com/ascend/aot)和[下载链接](https://ascend-package.obs.cn-north-4.myhuaweicloud.com:443/mindstudio-operator-tools/MindStudio-Operator-Debug-VSCode-Plugin-0.0.1.vsix)

### 特性
* 断点调试（设置/删除/禁用/启用断点）
* 单步执行调试(逐行执行/内部执行/跳出函数)/继续/暂停/重启
* 查看变量/寄存器/堆栈/断点信息，监视器支持执行表达式
* 内存查询
* 核切换

### 规格要求

* VSCode (1.88版本及以上)
* VSCode已安装远程登录插件和Hex Editor插件
* 支持平台: Linux
* 调试昇腾算子程序时，Linux环境已安装CANN工具包（含算子调试器）

### 约束
* 参考CANN工具包的约束要求

### 功能

#### 快速使用


1.  打开VSCode IDE界面，安装远程登录插件(如Remote-SSH)
2.  IDE界面远程SSH登录算子开发环境，打开已编译的算子工程文件
3.  安装[Hex Editor](https://marketplace.visualstudio.com/items?itemName=ms-vscode.hexeditor)插件和[MindStudio Operator Debug VSCode Plugin](https://ascend-package.obs.cn-north-4.myhuaweicloud.com:443/mindstudio-operator-tools/MindStudio-Operator-Debug-VSCode-Plugin-0.0.1.vsix)插件

* IDE界面插件市场界面离线本地安装
```
 - 插件.vsix文件上传linux环境
 - 展开IDE界面左侧边栏的插件菜单
 - 点击右上角...图标，选择Install from VSIX...选项，选择本地的插件文件进行安装
```

* IDE界面终端命令行离线本地安装
```
 - 插件.vsix文件上传linux环境
 - IDE界面点开终端命令行
 - 输入命令安装插件文件: code --install-extension /xx_dir/MindStudio-Operator-Debug-VSCode-Plugin-0.0.1.vsix
```
4.  点开debug侧边工具栏，若未生成launch.json文件，则根据提示点击生成launch.json文件(弹窗中调试器选择MSDebug)
5.  在`.vscode/launch.json`中添加调试配置(格式参考下面推荐配置)
6.  debug侧边工具栏中，选择已配置的调试器，点击`debug`或按`F5`启动调试

- **launch方式推荐配置**
```
{
    "configurations": [
        {
            "name": "xx-debug",
            "type": "msdebug-mi",
            "request": "launch",
            "cwd": "${workspaceFolder}",
            "target": "${workspaceFolder}/xx_operator",
            "msdebugMiPath": "/xx/x/msdebug-mi",
            "environmentScripts": [
                "/xxx/xx/Ascend/ascend-toolkit/set_env.sh"
            ],
            "environment": [{
                "name": "LD_LIBRARY_PATH",
                "value": "/xx/x/lib:${LD_LIBRARY_PATH}"
            }]
        }
    ]
}
```
* `name` 调试插件的名称
* `type` 固定值`"msdebug-mi"`
* `request` 固定值`"launch"`
* `cwd` 调试器启动的工作目录
* `target` (必要配置项)被调试的算子可执行文件的路径
* `msdebugMiPath` 调试器的路径
* `environmentScripts` 包含设置调试器的环境变量的脚本数组(脚本每行形如:`"export XX_KEY=XX_VALUE"`)，依次加载数组中各脚本
* `environment` 包含自定义环境变量的对象数组，在加载完`environmentScripts`脚本中的环境变量后依次加载`environment`中的环境变量，格式需满足:`[{"name": "xxx", "value":"yyy"}]`

#### 断点调试（设置/删除/禁用/启用断点）
* 可以在算子程序行号显示处设置、删除、禁用、用断点，也可以在debug侧边工具栏中的底部断点工具栏执行相同操作。

#### 单步执行调试(逐行执行/内部执行/跳出函数)/继续/暂停/重启
* 可以单击顶部调试工具栏上的图标控制程序，包括单步执行、步入、步出、继续、暂停、重启或停止程序的操作。

#### 查看变量/寄存器/堆栈/断点信息，监视器支持执行表达式
* 启动调试后，程序会停止在断点，当进程处于停止状态时，可以在IDE界面左侧查看当前线程的变量、堆栈、监视器和断点信息。其中监视器支持执行表达式。

#### 内存查询
* 查看变量的内存需要预先安装`Hex Editor`插件
* 调试停在断点时，光标移到变量右侧显示`0110`的内存按钮，点击按钮弹窗查看变量地址对应的内存值

#### 核切换
* 调试算子程序Kernel侧代码时，IDE右下角显示当前调试占用NPU卡的核ID，点击该按钮（形如:`kernel:aiv 10`），弹窗显示**所有可用核ID**，根据提示将**待切核ID**输入弹窗并回车进行切核，输入格式例如`aiv 15`
