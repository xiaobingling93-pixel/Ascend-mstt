# 迁移工具快速入门样例介绍

## 说明

本样例选用ResNet50模型。

## 环境准备

1. 准备一台基于Atlas 训练系列产品的训练服务器，并[安装NPU驱动和固件](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0005.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)。

2. 安装开发套件包Ascend-cann-toolkit，具体请参考[安装CANN软件包](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)。

3. 以安装PyTorch 2.1.0版本为例，具体操作请参考适配插件开发（PyTorch框架)。

4. 配置环境变量。<br>
安装CANN软件后，使用CANN运行用户进行编译、运行时，需要以CANN运行用户登录环境，执行source ${install_path}/set_env.sh命令设置环境变量。其中${install_path}为CANN软件的安装目录，例如：/usr/local/Ascend/ascend-toolkit。

5. 下载[main.py](main.py)文件，并上传至训练服务器的个人目录下。


## 执行迁移
1. 在训练脚本（main.py文件）中导入自动迁移的库代码。
```Python
 42
 43 import torch_npu
 44 from torch_npu.contrib import transfer_to_npu
 45
# 在43 44行插入的代码为自动迁移的库代码，可以在NPU环境下直接执行训练
```

2. 迁移完成后的训练脚本可在NPU上运行，执行以下训练命令。
```Python
python main.py -a resnet50 -b 32 --gpu 1 --dummy
```
如果训练正常进行，开始打印迭代日志，说明训练功能迁移成功，如下所示。
```Python
Use GPU: 1 for training
=> creating model 'resnet50'
=> Dummy data is used!
Epoch: [0][    1/40037] Time  8.287 ( 8.287)    Data  0.504 ( 0.504)    Loss 7.0919e+00 (7.0919e+00)    Acc@1   0.00 (  0.00)   Acc@5   0.00 (  0.00)
Epoch: [0][   11/40037] Time  0.097 ( 1.268)    Data  0.000 ( 0.479)    Loss 1.5627e+01 (1.8089e+01)    Acc@1   0.00 (  0.00)   Acc@5   3.12 (  0.57)
Epoch: [0][   21/40037] Time  0.096 ( 0.710)    Data  0.000 ( 0.253)    Loss 7.7462e+00 (1.4883e+01)    Acc@1   0.00 (  0.00)
```

3. 成功保存权重，说明保存权重功能迁移成功。
