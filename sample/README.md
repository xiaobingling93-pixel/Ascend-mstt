# 样例库介绍

## 说明
本案例库主要用于配合AscendC算子开发工具的能力演示，所以，算子工程做了深度简化，聚焦于辅助工具展示。

如果考虑商用集成，推荐使用CANN软件包中的AscendC样例工程，比如：ascendc_kernel_cmake目录。本项目中的工程就是基于其进行简化仅用于快速验证。

## 依赖说明
安装CANN包，并使能环境变量，并确保```ASCEND_HOME_PATH```生效，可以在CANN包安装目录下使能：
```
source set_env.sh
```

## 目录说明
```normal_sample```目录中主要是算子样例，内部分为aic，aiv，mix的算子样例，支持调试和调优使用。
```sanitizer```目录中主要是各类异常样例，用于体现异常检测工具的能力。

## 其他
现msprof使能仿真时，还需要额外调整，具体如下：
1. 编译阶段：在```sample\normal_sample\vec_only```相对路径下的```Makefile```文件中修改如下内容：
   + 仿真器依赖：
    ```
    LINK_LIBS					:= -L${ASCEND_HOME_PATH}/lib64 -lruntime -lascendcl -lstdc++
    修改为：
    LINK_LIBS					:= -L${ASCEND_HOME_PATH}/lib64 -L${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib/ -lruntime_camodel -lascendcl -lstdc++  # 需要添加libruntime_camodel的依赖路径, SOC_VERSION 使用npu-smi info查询NPU Name
    ```
    + 调试信息增强：
    ```
    COMPILER_FLAG		:= -xcce -O2 -std=c++17
    修改为：
    COMPILER_FLAG		:= -xcce -O2 -std=c++17 -g
    ```

2. 运行阶段：添加依赖库路径，
  ```
  export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib/:$LD_LIBRARY_PATH  # SOC_VERSION 使用npu-smi info查询NPU Name，如：名字为910A，则填入：Ascend910A
  msprof op simulator --application=./add.fatbin # 在对应路径下
  ```

现msdebug进行上板调试时，还需要额外调整，具体如下：
1. 编译阶段：在```sample\normal_sample\vec_only```相对路径下的```Makefile```文件中修改如下内容：
    + 调试信息增强，并扩大栈空间：
    ```
    COMPILER_FLAG		:= -xcce -O2 -std=c++17
    修改为：
    COMPILER_FLAG		:= -xcce -O0 -std=c++17 -g -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-jump-expand=true