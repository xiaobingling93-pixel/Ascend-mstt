# Ascend Extension for dynolog

## 安装方式

### 1. clone 代码

```bash
git clone https://gitee.com/ascend/mstt.git
```

### 2. 安装依赖
dynolog的编译依赖，确保安装了以下依赖：
<table>
  <tr>
   <td>Language
   </td>
   <td>Toolchain
   </td>
  </tr>
  <tr>
   <td>C++
   </td>
   <td>gcc 8.5.0+
   </td>
  </tr>
  <tr>
   <td>Rust
   </td>
   <td>Rust 1.58.1 (1.56+ required for clap dependency)
   </td>
  </tr>
</table>

- 安装rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

source $HOME/.cargo/env
```

- 安装ninja

```bash
# debian
sudo apt-get install -y cmake ninja-build

# centos
sudo yum install -y cmake ninja
```

### 3. 编译

默认编译生成dyno和dynolog二进制文件, -t参数可以支持将二进制文件打包成deb包或rpm包.

```bash
# 编译dyno和dynolog二进制文件
bash scripts/build.sh

# 编译deb包, 当前支持amd64和aarch64平台, 默认为amd64, 编译aarch64平台需要修改third_party/dynolog/scripts/debian/control文件中的Architecture改为aarch64
bash scripts/build.sh -t deb

# 编译rpm包, 当前只支持amd64平台
bash scripts/build.sh -t rpm
```

## 使用方式

### Profiler trace dump功能
Profiler trace dump功能基于dynolog开发，实现类似于动态profiling的动态触发Ascend Torch Profiler采集profiling的功能。用户基于dyno CLI命令行可以动态触发指定节点的训练进程trace dump。

- 查看nputrace支持的命令和帮助

```bash
dyno nputrace --help
```

- nputrace使用方式

```bash
dyno nputrace [SUBCOMMANDS] --log-file <LOG_FILE>
```

nputrace子命令支持的参数选项

| 子命令 | 参数类型 | 说明 |
|-------|-------|-------|
| record_shapes | action | 是否采集算子的InputShapes和InputTypes，设置参数采集，默认不采集 |
| profile_memory | action | 是否采集算子内存信息，设置参数采集，默认不采集 |
| with_stack | action | 是否采集Python调用栈，设置参数采集，默认不采集 |
| with_flops | action | 是否采集算子flops，设置参数采集，默认不采集 |
| with_modules | action | 是否采集modules层级的Python调用栈，设置参数采集，默认不采集 |
| analyse | action | 采集后是否自动解析，设置参数解析，默认不解析 |
| l2_cache | action | 是否采集L2 Cache数据，设置参数采集，默认不采集 |
| op_attr | action | 是否采集算子属性信息，设置参数采集，默认不采集 |
| data_simplification | String | 解析完成后是否数据精简，可选值范围[`true`, `false`]，默认值`true` |
| activities | String | 控制CPU、NPU事件采集范围，可选值范围[`CPU,NPU`, `NPU,CPU`, `CPU`, `NPU`]，默认值`CPU,NPU` |
| profiler_level | String | 控制profiler的采集等级，可选值范围[`Level_none`, `Level0`, `Level1`, `Level2`]，默认值`Level0`|
| aic_metrics | String | AI Core的性能指标采集项，可选值范围[`AiCoreNone`, `PipeUtilization`, `ArithmeticUtilization`, `Memory`, `MemoryL0`, `ResourceConflictRatio`, `MemoryUB`, `L2Cache`, `MemoryAccess`]，默认值`AiCoreNone`|
| export_type | String | profiler解析导出数据的类型，可选值范围[`Text`, `Db`]，默认值`Text`|
| gc_detect_threshold | Option<f32> | GC检测阈值，单位ms，只采集超过阈值的GC事件。该参数为可选参数，默认不设置时不开启GC检测 |

- nputrace示例命令

```bash
# 示例1：采集框架、CANN和device数据，同时采集完后自动解析以及解析完成不做数据精简，落盘路径为/tmp/profile_data
dyno nputrace --activities CPU,NPU --analyse --data_simplification false --log-file /tmp/profile_data

# 示例2：只采集CANN和device数据，同时采集完后自动解析以及解析完成后开启数据精简，落盘路径为/tmp/profile_data
dyno nputrace --activities NPU --analyse --data_simplification true --log-file /tmp/profile_data

# 示例3：只采集CANN和device数据，只采集不解析，落盘路径为/tmp/profile_data
dyno nputrace --activities NPU --log-file /tmp/profile_data
```
