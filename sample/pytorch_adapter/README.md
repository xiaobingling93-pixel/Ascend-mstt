# 自定义算子的pytorch框架适配说明

## 简介
昇腾提供丰富的算子接入框架的方式，此处将介绍最简单的一种，每个目录中都是一个独立的可使用的工程

## 依赖
与业内pytorch的算子介入方式相同，算子接入框架需要保障设备上有正确的pytorch版本（我们还依赖torch_npu版本）

pytorch版本可由pip安装，torch_npu版本详见[此处](https://gitee.com/ascend/pytorch/releases)，请选择与pytorch适配的torch_npu版本。

## 工程介绍
整体工程目录如下：
```
- pytorch_adapter
  |- jit_compile         # 实时编译的接入方式
    |- add_adapter.cpp   # 使用算子动态库接口完成算子在pytorch框架的适配
    |- add_kernel.cpp    # 昇腾算子实现，并提供host侧的动态库接口
    |- main.py           # python的入口，实现整体集成
    |- Makefile          # 用以生成昇腾算子的host侧动态库的编译脚本
  |- with_setuptools     # wheel包的接入方式
    |- add_adapter.cpp
    |- add_kernel.cpp
    |- Makefile
    |- setup.py          # setuptools的入口，支持编译并打包生成wheel包
    |- test.py           # 测试wheel包功能的入口
```

## 工程使用

### jit_compile工程
执行如下命令，就会在运行过程中，现场生成python模块并使用：
```
python main.py
```

### setuptools工程
针对with_setuptools工程，可以编译出可安装的wheel包，便于多机部署使用。


1. 执行如下命令可以编译出软件包(setuptools可以支持多种方式，比如：build,install等，此处不一一展示)：
```
pytorch setup.py bdist_wheel    # 编译出wheel包，在dist目录下
```

2. 到```dist```目录下用pip命令安装对应软件包。

3. 执行测试脚本
```
python test.py
```

## 其他
1. 此处样例使用的是静态tiling，如果使用动态tiling，则可以在adapter.cpp中对Tensor的shape进行分析，选择合适tiling。（这部分是流程中必须的，只是可能在不同位置，比如aclnn中，这部分在接口实现；此处，我们本身也可以对add_custom_do进行封装，将tiling内置。）