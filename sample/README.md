# 样例库介绍

## 说明
本案例库主要用于配合AscendC算子开发工具的能力演示，所以，算子工程做了深度简化，聚焦于辅助工具展示。

如果考虑商用集成，推荐使用CANN软件包中的AscendC样例工程，比如：ascendc_kernel_cmake目录。本项目中的工程就是基于其进行简化仅用于快速验证。

说明：该sample目录中，每个最小目录就是一个完整的样例工程。这些样例工程本身可能以为依赖的不同存在差异。

## 依赖说明
- 硬件环境请参见《[昇腾产品形态说明](https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/quickstart/quickstart/quickstart_18_0002.html)》。
- 软件环境请参见《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/softwareinst/instg/instg_0000.html?Mode=PmIns&OS=Debian&Software=cannToolKit)》安装昇腾设备开发或运行环境，即toolkit软件包。

以上环境依赖请根据实际环境选择适配的版本。

### 版本配套   
| 条件 | 要求 |
|---|---|
| CANN版本 | >=8.0.RC1.alpha001 |
| 硬件要求 | Atlas 800T A2 训练服务器|

- 支持AscendPyTorch 1.11.0或更高版本，支持的PyTorch和CANN以及PyTorch和Python软件版本配套关系请参见《[Ascend Extension for PyTorch插件](https://gitcode.com/Ascend/pytorch)》。
- 固件驱动版本与配套CANN软件支持的固件驱动版本相同，开发者可通过“[昇腾社区-固件与驱动](https://www.hiascend.com/hardware/firmware-drivers/community%3Fproduct=2)”页面根据产品型号与CANN软件版本获取配套的固件与驱动。

## 目录介绍
整体目录结构如下：
```
- sample
  |- build              # 编译并运行所有样例内容（建议按需使用，此处命令可以参考
  |- normal_sample      # 纯C/C++的AscendC单算子极简工程，可配合msdebug和msprof工具
    |- cube_only        # 仅含aic的AscendC单算子极简工程
    |- mix              # mix算子的AscendC单算子极简工程
    |- vec_only         # 仅含aiv的AscendC单算子极简工程
  |- pytorch_adapter    # 适配pytorch的AscendC单算子极简工程，可配合msdebug和msprof工具
    |- jit_compile      # jit模式，运行时编译使用
    |- with_setuptools  # 编译成wheel包安装使用
  |- sanitizer_sample   # 异常样例，用于配合mssanitizer工具
    |- racecheck        # 含竞争问题的样例
    |- xx               # 其他异常样例 
```

如果你关注自定义算子的pytorch框架适配，详见[此处](./pytorch_adapter/README.md)


## 算子调试 msdebug
若使用msdebug进行上板调试，还需要额外调整，具体如下：
1. 编译阶段：在```sample\normal_sample\vec_only```相对路径下的```Makefile```文件中修改如下内容：
    + 调试信息增强，并扩大栈空间：
    ```
    COMPILER_FLAG		:= -xcce -O2 -std=c++17
    修改为：
    COMPILER_FLAG		:= -xcce -O0 -std=c++17 -g --cce-ignore-always-inline=true
    ```

2. 运行阶段：
```
msdebug ./*.fatbin
```

## 内存检测 sanitizer
1. 编译阶段：在编译过程中添加```--cce-enable-sanitizer -g```参数, 在链接过程中添加```--cce-enable-sanitizer```参数。（现样例中已在Makefile中添加），执行如下命令：
```
make
```

2. 运行阶段：
```
mssanitizer ./*.fatbin  # 默认进行memcheck检查
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
 LINK_LIBS					:= -L${ASCEND_HOME_PATH}/lib64 -L${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib/ -lruntime_camodel -lascendcl -lstdc++  # 需要添加libruntime_camodel的依赖路径, SOC_VERSION 通过使用npu-smi info命令进行查询，获取Chip Name信息。实际配置值                                                                                                                                                     为AscendChip Name，例如Chip Name取值为xxxyy，实际配置值为Ascendxxxyy。当Ascendxxxyy为代码样例路径时，需要配置ascendxxxyy。
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
    # SOC_VERSION为NPU名称，可通过npu-smi info命令进行查询。
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
