# 样例库介绍

## 说明
本案例库主要用于配合AscendC算子开发工具的能力演示，所以，算子工程做了深度简化，聚焦于辅助工具展示。

如果考虑商用集成，推荐使用CANN软件包中的AscendC样例工程，比如：ascendc_kernel_cmake目录。本项目中的工程就是基于其进行简化仅用于快速验证。

## 依赖说明
安装CANN包，并使能环境变量，并确保```ASCEND_HOME_PATH```生效，可以在CANN包安装目录下使能：
```
source set_env.sh
```

## 算子调优
算子调优工具可以支持上板和仿真算子的调优，下面将以vec_only中的算子为例，进行工具使用的实战命令讲解

### 上板调优
1. 基于原始的sample代码，无需修改，直接编译算子，获得add.fatbin
    ```
    cd ./sample/normal_sample/vec_only
    make clean && make
    ```
2. 使用算子调优工具，对算子程序进行调优。`--aic-metrics`省略，使用默认全部开启；`--output`参数省略，使用默认值`./`
    ```
    msprof op --application=./add.fatbin
    ```
3. 在当前目录下可以看到`OPPROF_`开头的文件夹，进入后将包含开启的`--aic-metrics`开关对应的csv数据和算子基础数据`OpBasicInfo.csv`。查看对应的csv文件即可获得算子的block级别的性能数据。（当前的算子性能数据是算子预热后的数据）
    ```
    OPPROF_2024xxxx_XXXXXX
    ├── dump
    ├── OpBasicInfo.csv
    ├── ArithmeticUtilization.csv
    ├── ... (开启的aic-metrics)
    └──  ResourceConflictRatio.csv
    ```
4. 更多csv中指标信息请参考算子开发工具使用手册。

### 仿真调优
使用msprof进行仿真调优时，需要编译出可以运行在仿真器上的可执行算子，需要对编译选项稍作修改，修改如下
在```./sample/normal_sample/vec_only```相对路径下的```Makefile```文件中修改如下内容：
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

下面将从编译阶段开始进行

1. 仿真算子编译
   ```
    cd ./sample/normal_sample/vec_only
    make clean && make
   ```
2. 添加运行时依赖库路径
    ```
    # SOC_VERSION 使用npu-smi info查询NPU Name，如：名字为910A，则填入：Ascend910A
    export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib/:$LD_LIBRARY_PATH  
    ```
3. 使用算子调优工具进行仿真调优，获取仿真性能数据，`--output`参数省略，使用默认值`./`
   ```
   msprof op simulator --application=./add.fatbin
   ```
4. 在当前目录下可以看到`OPPROF_`开头的文件夹，，生成以OPPROF_时间_随机字符串的文件夹，结构如下：
    ```
    OPPROF_20231023120542_FQMZMGOHUYVUZEXP
    ├── dump                                    # 原始性能数据，无需关注
    └── simulation                              # 仿真性能数据分析结果
        ├── core0.veccore0                      # 算子block级子核，vec样例中使用了8个
            ├── core0.veccore1_code_exe.csv     # 代码行耗时
            ├── core0.veccore1_instr_exe.csv    # 程序代码指令详细信息
            └── trace.json                      # 算子block级子核流水图
        ├── ...
        ├── api                                 # 算子热点图文件夹，将文件夹内全部文件拖入Ascend Compute即可
            ├── api.json                        # 代码热点映射
            └── <user-kernel>.cpp               # 算子kernel代码
        ├── visualize_data.bin                  # 算子可视化文件，使用Ascend Insight加载
        └── trace.json                          # 算子所有核的流水图
    ```
4. 更多指标信息请参考算子开发工具使用手册。

## 算子调试msdebug
若使用msdebug进行上板调试，还需要额外调整，具体如下：
1. 编译阶段：在```sample\normal_sample\vec_only```相对路径下的```Makefile```文件中修改如下内容：
    + 调试信息增强，并扩大栈空间：
    ```
    COMPILER_FLAG		:= -xcce -O2 -std=c++17
    修改为：
    COMPILER_FLAG		:= -xcce -O0 -std=c++17 -g -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-jump-expand=true

## 内存检测 sanitizer
### sanitizer_sample目录介绍

此目录下为sanitizer对应的样例库，包含竞争检测和内存检测相关的样例。

#### Racecheck目录介绍

Racecheck为竞争检测相关的样例。

raw_error_kernel.cpp文件为UB上先读后写竞争和GM上先写后读竞争问题的样例。


运行阶段：

```
/usr/local/Ascend/ascend-toolkit/latest/tools/mssanitizer/bin/mssanitizer --tool=racecheck ./raw_error.fatbin
```