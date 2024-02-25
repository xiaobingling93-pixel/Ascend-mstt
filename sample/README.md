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
现仅支持```sample\normal_sample\vec_only```样例。

大概在2月份会补齐各类工具样例和文档

### sanitizer_sample目录介绍

此目录下为sanitizer对应的样例库，包含竞争检测和内存检测相关的样例。

#### Racecheck目录介绍

Racecheck为竞争检测相关的样例。

raw_error_kernel.cpp文件为UB上先读后写竞争和GM上先写后读竞争问题的样例。


运行阶段：

```
/usr/local/Ascend/ascend-toolkit/latest/tools/mssanitizer/bin/mssanitizer --tool=racecheck ./raw_error.fatbin
```

## 其他
现msprof使能仿真时，还需要额外调整，具体如下：
1. 编译阶段：在```sample\normal_sample\vec_only```相对路径下的```Makefile```文件中修改如下内容：
   + 仿真器依赖：
    ```
    LINK_LIBS			:= -L${TOP_DIR}/third_party/lib/ -lruntime -lascendcl -lstdc++
    修改为：
    LINK_LIBS			:= -L${TOP_DIR}/third_party/lib/ -lruntime_camodel -lascendcl -lstdc++
    ```
    + 调试信息增强：
    ```
    COMPILER_FLAG		:= -xcce -O2 -std=c++17
    修改为：
    COMPILER_FLAG		:= -xcce -O2 -std=c++17 -g
    ```

2. 运行阶段：添加依赖库路径，
  ```
  export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/aarch64-linux/simulator/${SOC_VERSION}/lib/:$LD_LIBRARY_PATH  # 使用npu-smi info查询NPU Name，如：名字为910A，则填入：Ascend910A
  msprof op simulator --application=./add.fatbin # 在对应路径下
  ```
