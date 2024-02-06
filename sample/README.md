# 样例库介绍

## 说明
本案例库主要用于配合AscendC算子开发工具的能力演示，所以，对于整体算子工程做了一定的简化，聚焦于辅助工具展示。

## 依赖说明
安装CANN包，并使能环境变量，并确保```ASCEND_HOME_PATH```生效，可以在CANN包安装目录下使能
```
source set_env.sh
```

## 目录说明
现仅支持```sample\normal_sample\vec_only```样例。
大概在2月份会补齐各类工具案例和文档

## 其他
现msprof使能仿真时，还需要额外调整，具体如下：
1. 修改编译脚本，在```sample\normal_sample\vec_only```相对路径下的```Makefile```文件中修改如下内容：
  a. 修改依赖，支持仿真器能力
```
LINK_LIBS			:= -L${TOP_DIR}/third_party/lib/  -lruntime -lascendcl -lstdc++
修改为：
LINK_LIBS			:= -L${TOP_DIR}/third_party/lib/  -lruntime_camodel -lascendcl -lstdc++
```
  b. 添加调试信息，支持扩展能力
```
COMPILER_FLAG		:= -xcce -O2 -std=c++17 --cce-aicore-arch=dav-c220-vec
修改为：
COMPILER_FLAG		:= -xcce -O2 -std=c++17 --cce-aicore-arch=dav-c220-vec -g
```

2. 运行时添加依赖库路径
```
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/aarch64-linux/simulator/Ascend910B1/lib/:$LD_LIBRARY_PATH
msprof op simulator --application=./add.fatbin # 在对应路径下
```
