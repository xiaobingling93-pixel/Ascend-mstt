# **数据解析工具**

数据解析工具（parse_tool）提供命令行交互式界面，提供更多的数据解析功能并且展示结果。

使用场景：本工具主要用于精度比对前后两次NPU kernel层级dump数据的一致性。

## 进入parse交互式界面

安装atat工具后（详见《[MindStudio精度调试工具](../../README.md)》的“工具安装”章节），可以通过使用命令 **atat -f pytorch parse** 进入交互式界面，如下所示：

```bash
atat -f pytorch parse
Parse >>>
```

可在parse的界面中执行Shell命令，以及如下场景的相关解析命令：

- 支持指定kernel层级算子数据比对。
- 支持指定kernel层级算子数据转换及展示。
- 支持交互式指定pkl文件中API对应dump数据查看。
- 支持API进行可选层级比对和打印（统计级和像素级）。

Ctrl+C可以退出parse交互式界面。不退出parse交互式界面若需要执行非该界面下的内置Shell命令，且命令与parse交互式界面命令冲突时，非该界面命令需要使用run命令，在相关命令前加上run前缀，如下示例：

```bash
atat -f pytorch parse
Parse >>> run vim cli.py
Parse >>> vim cli.py
```

以上各场景详细介绍请参见下文章节。

## kernel层级算子数据批量转换

本功能会将原有待比对dump数据目录下的dump数据按照算子名和时间戳进行梳理并分类，之后再将dump数据转为为npy文件。

依赖：CANN包中的msaccucmp工具，需要安装Ascend-CANN-toolkit，详见《[CANN 软件安装指南](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2Fcanncommercial%2F700%2Fenvdeployment%2Finstg%2Finstg_0001.html)》。

输入以下比对命令进行数据转换。

```bash
cad -m my_dump_path [-out output_path] [-asc msaccucmp_path]
```

| 参数名称 | 说明                                                         | 是否必选 |
| -------- | ------------------------------------------------------------ | -------- |
| -m       | 待转换kernel dump数据目录。需要指定到kernel dump数据的deviceid级目录。 | 是       |
| -out     | 结果输出目录，须指定已存在的目录，默认为./parse_data/acl_batch_convert。未指定时保存在默认路径下，比对结束后会打印log提示输出结果存放路径。 | 否       |
| -asc     | 指定msaccucmp路径，默认路径为：/usr/local/Ascend/ascend-toolkit/latest/tools/operator_cmp/compare/msaccucmp.py。 | 否       |

**示例**

```
# 传入待比对数据目录
Parse >>> cad -m /home/xxx/my_dump_path/20000124003856/0
# 转换结果打印
......
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────╮
# 转换前的dump文件
│ SrcFile: /home/xxx/my_dump_path/20000124003856/0/272/TransData.trans_TransData_22.112.21.948645536672764 │
# 转换后的npy文件
│ - TransData.trans_TransData_22.112.21.948645536672764.output.0.npy                                       │
│ - TransData.trans_TransData_22.112.21.948645536672764.input.0.npy                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────╯
......
[INFO] The comparison result have been written to "./parse_data/acl_batch_convert".
```

输出结果：

原dump数据目录：

```
├── /home/xxx/my_dump_path/20000124003856/0/
│   ├── 272
│   │   ├── {op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}
│   │   ...
│   ├── 512
│   ...
```

转换后：

```
├── ./parse_data/acl_batch_convert/{timestamp}
│   ├── {op_name1}
│   │   ├── {timestamp1}
│   │   |   ├── {op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}.{input/output}.{参数序号}.npy
│   │   |   │   ...
│   │   ├── {timestamp2}
│   │   |   ...
│   ├── {op_name2}
│   ├── ...
```

## kernel层级算子数据比对

本功能主要用于比对前后两次NPU kernel层级dump数据的一致性。

本功能支持批量比对，若需要进行批量比对，需要先将两份待比对的NPU kernel层级dump数据进行“**kernel层级算子数据批量转换**”，可以使两份数据更好的匹配；若直接进行dump数据的比对，建议只比对单个dump数据文件。

输入以下比对命令进行数据比对。

```bash
vc -m my_dump_path -g golden_dump_path [-out output_path] [-cmp_path msaccucmp_path]
```

| 参数名称  | 说明                                                         | 是否必选 |
| --------- | ------------------------------------------------------------ | -------- |
| -m        | 待比对kernel dump数据目录。如果比对单个算子，需要指定到kernel dump数据的model_id级目录；如果批量比对，则指定到cad转换后的timestamp级目录。 | 是       |
| -g        | 标杆kernel dump数据目录。如果比对单个算子，需要指定到kernel dump数据的model_id级目录；如果批量比对，则指定到cad转换后的timestamp级目录。 | 是       |
| -out      | 结果输出目录，须指定已存在的目录，默认为./parse_data/acl_batch_comapre。未指定时保存在默认路径下，比对结束后会打印log提示输出结果存放路径。 | 否       |
| -cmp_path | 指定msaccucmp路径，默认路径为：/usr/local/Ascend/ascend-toolkit/latest/tools/operator_cmp/compare/msaccucmp.py | 否       |

输出结果：batch_compare_{timestamp}.csv文件。

**示例**

```bash
# 传入待比对数据目录以及标杆数据目录
Parse >>> vc -m ./my_dump_path -g ./golden_data_path
[INFO]Compare result is saved in : parse_data/acl_batch_comapre/batch_compare_1707271118.csv
```

## kernel算子数据的npy转换

依赖：CANN包中的msaccucmp工具，需要安装Ascend-CANN-toolkit，详见《[CANN 软件安装指南](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2Fcanncommercial%2F700%2Fenvdeployment%2Finstg%2Finstg_0001.html)》。

输入以下转换命令进行数据转换， 将kernel级别dump数据转为npy文件。

```bash
dc -n file_name/file_path [-f format] [-out output_path]
```

| 参数名称  | 说明                                                         | 是否必选 |
| --------- | ------------------------------------------------------------ | -------- |
| -n        | 需转换的dump数据文件或dump数据文件目录。                     | 是       |
| -f        | 开启format转换，指定该参数时需要配置format格式。当前内置的Format转换支持如下类型： FRACTAL_NZ转换NCHW FRACTAL_NZ转换成NHWC FRACTAL_NZ转换ND HWCN转换FRACTAL_Z HWCN转换成NCHW HWCN转换成NHWC NC1HWC0转换成HWCN NC1HWC0转换成NCHW NC1HWC0转换成NHWC NCHW转换成FRACTAL_Z NCHW转换成NHWC NHWC转换成FRACTAL_Z NHWC转换成HWCN NHWC转换成NCHW NDC1HWC0转换成NCDHW | 否       |
| -out      | 结果输出目录。                                               | 否       |
| -cmp_path | 指定msaccucmp路径，默认路径为：/usr/local/Ascend/ascend-toolkit/latest/tools/operator_cmp/compare/msaccucmp.py | 否       |

- 输出结果：npy文件。

- 若指定-out参数需要用户传入输出路径，并且路径需要已存在。

- 若未指定输出目录， 则比对结束后将结果保存在默认目录 “./parse_data/convert_result”中，比对结束后会打印log提示输出结果存放路径及转换结果。

- 输入以下命令，展示npy数据统计信息。

  ```bash
  pt -n file_path
  ```

  | 参数名称 | 说明          | 是否必选 |
  | -------- | ------------- | -------- |
  | -n       | npy文件路径。 | 是       |

  打印统计信息：shape, dtype, max, min和mean。默认在npy文件路径下将该数据保存为txt文件。

**示例1**

```bash
# 传入需转换的dump文件目录
Parse >>> dc -n ./dump_data/
......
# 转换结果
╭──────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ SrcFile: ./dump_data/
│  - Add.fp32_vars_add_2fp32_vars_Relu_9.31.5.1636595794731103.input.0.npy                             │
│  - Add.fp32_vars_add_1fp32_vars_Relu_6.24.5.1636595794631347.output.0.npy                            │
│  - Add.fp32_vars_add_2fp32_vars_Relu_9.31.5.1636595794731103.input.1.npy                             │
│  - Add.fp32_vars_add_1fp32_vars_Relu_6.24.5.1636595794631347.input.1.npy                             │
│  - Add.fp32_vars_add_3fp32_vars_Relu_12.40.5.1636595794846124.input.1.npy                            │
│  - Add.fp32_vars_add_1fp32_vars_Relu_6.24.5.1636595794631347.input.0.npy                             │
│  - Add.fp32_vars_add_3fp32_vars_Relu_12.40.5.1636595794846124.input.0.npy                            │
│  - Add.fp32_vars_add_2fp32_vars_Relu_9.31.5.1636595794731103.output.0.npy                            │
│  - Add.fp32_vars_add_3fp32_vars_Relu_12.40.5.1636595794846124.output.0.npy                           │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

**示例2**

```bash
# 查看某个dump数据块的数据信息
# 默认会将数据中的tensor保存成 txt
Parse >>> pt -n ./parse_data/dump_convert/Add.fp32_vars_add_1fp32_vars_Relu_6.24.5.1636595794631347.output.0.npy
......
# 打印统计信息
[Shape: (1, 16, 56, 56, 16)] [Dtype: float16] [Max: 452.0] [Min: -408.5] [Mean: -3.809]
Path: ./parse_data/dump_convert/Add.fp32_vars_add_1fp32_vars_Relu_6.24.5.1636595794631347.input.0.npy                           
TextFile:./parse_data/dump_convert/Add.fp32_vars_add_1fp32_vars_Relu_6.24.5.1636595794631347.input.0.npy.txt
```

## dump.json文件中指定API的dump数据信息查看（暂不支持）

输入以下命令，解析并输出dump.json文件中指定API的统计信息。

```bash
pk -f pkl_path -n api_name
```

| 参数名称 | 说明                    | 是否必选 |
| -------- | ----------------------- | -------- |
| -f       | 指定dump.json文件路径。 | 是       |
| -n       | 指定API名称。           | 是       |

- 输出结果：打印统计信息（shape, dtype, max和min mean）。
- 若pkl文件中存在相应的堆栈信息，则会打印堆栈信息。

**示例**

```bash
# 传入pkl文件及api名称
Parse >>> pk -f ./torch_dump/xxx/rank0/dump.json -n Functional_conv2d_0_forward
......
# 打印统计信息及堆栈（pkl文件不包含堆栈则不会打印堆栈）

Statistic Info:
  [Functional_conv2d_0_forward_input.0][dtype: torch.float32][shape: [2, 1, 2, 2]][max: 1.576936960220337][min: -0.9757485389709473][mean: 0.4961632490158081]
  [Functional_conv2d_0_forward_input.1][dtype: torch.float32][shape: [2, 1, 2, 2]][max: 0.20064473152160645][min: -0.47102075815200806][mean: -0.20796933770179749]
  [Functional_conv2d_0_forward_input.2][dtype: torch.float32][shape: [2]][max: 0.17380613088607788][min: -0.16853803396224976][mean: 0.0026340484619140625]
  [Functional_conv2d_0_forward_output][dtype: torch.float32][shape: [2, 2, 1, 1]][max: 0.02364911139011383][min: -1.762906551361084][mean: -0.6710853576660156]
```

## API可选层级比对

输入以下命令, 进行统计级和像素级比对。

```bash
cn -m my_data*.npy -g gloden*.npy [-p num] [-al atol] [-rl rtol]
```

- 统计级比对：对tensor整体进行余弦值及相对误差的计算。
- 像素级比对：对输入的两个npy文件进行逐元素比对。若两个tensor对应元素的相对误差或绝对误差大于**误差阈值**（-al和-rl配置）则被标记为错误数据。

| 参数名称 | 说明                                            | 是否必选 |
| -------- | ----------------------------------------------- | -------- |
| -m       | 待比对数据。                                    | 是       |
| -g       | 标杆数据。                                      | 是       |
| -p       | 设置比对结束后打印错误元素的个数，默认值20。    | 否       |
| -al      | 判定数据存在精度问题的绝对误差阈值，默认0.001。 | 否       |
| -rl      | 判定数据存在精度问题的相对误差阈值，默认0.001。 | 否       |
| -s       | 将npy文件保存成txt文件，用于查看，默认开启。    | 否       |

输出结果：

- 统计级比对结果。
- 两个文件的统计信息（shape, dtype, max, min和mean）。
- 错误数据打印表格。

**示例**

```bash
# 对比两个tensor的数据
Parse >>> cn -m Add.InceptionV3_InceptionV3_Mixed_7a_Branch_0_add_3.323.1619494134703053.output.0.npy -g InceptionV3_InceptionV3_Mixed_7a_Branch_0_add_3.0.1619492699305998.npy -p 10 -s -al 0.002 -rl 0.005
                  Error Item Table                                        Top Item Table
┏━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓ ┏━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Index ┃ Left          ┃ Right        ┃ Diff         ┃ ┃ Index ┃ Left        ┃ Right       ┃ Diff          ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩ ┡━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ 155   │ 0.024600908   │ 0.022271132  │ 0.002329776  │ │ 0     │ -0.9206961  │ -0.9222216  │ 0.0015255213  │
│ 247   │ 0.015752593   │ 0.017937578  │ 0.0021849852 │ │ 1     │ -0.6416973  │ -0.64051837 │ 0.0011789203  │
│ 282   │ -0.0101207765 │ -0.007852031 │ 0.0022687456 │ │ 2     │ -0.35383835 │ -0.35433492 │ 0.0004965663  │
│ 292   │ 0.019581757   │ 0.02240482   │ 0.0028230622 │ │ 3     │ -0.18851271 │ -0.18883198 │ 0.00031927228 │
│ 640   │ -0.06593232   │ -0.06874806  │ 0.0028157383 │ │ 4     │ -0.43508735 │ -0.43534422 │ 0.00025686622 │
│ 1420  │ 0.09293677    │ 0.09586689   │ 0.0029301196 │ │ 5     │ 1.4447614   │ 1.4466647   │ 0.0019032955  │
│ 1462  │ -0.085207745  │ -0.088047795 │ 0.0028400496 │ │ 6     │ -0.3455438  │ -0.3444429  │ 0.0011008978  │
│ 1891  │ -0.03433288   │ -0.036525503 │ 0.002192624  │ │ 7     │ -0.6560242  │ -0.6564579  │ 0.0004336834  │
│ 2033  │ 0.06828873    │ 0.07139922   │ 0.0031104907 │ │ 8     │ -2.6964858  │ -2.6975214  │ 0.0010356903  │
│ 2246  │ -0.06376442   │ -0.06121233  │ 0.002552092  │ │ 9     │ -0.73746175 │ -0.73650354 │ 0.00095820427 │
└───────┴───────────────┴──────────────┴──────────────┘ └───────┴─────────────┴─────────────┴───────────────┘
╭───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Left:                                                                                                                                 |
│  |- NpyFile: ./dump/temp/decode/Add.InceptionV3_InceptionV3_Mixed_7a_Branch_0_add_3.323.1619494134703053.output.0.npy                 |
│  |- TxtFile: ./dump/temp/decode/Add.InceptionV3_InceptionV3_Mixed_7a_Branch_0_add_3.323.1619494134703053.output.0.npy.txt             |
│  |- NpySpec: [Shape: (32, 8, 8, 320)] [Dtype: float32] [Max: 5.846897] [Min: -8.368301] [Mean: -0.72565556]                           |
│ DstFile:                                                                                                                              │
│  |- NpyFile: ./dump/cpu/InceptionV3_InceptionV3_Mixed_7a_Branch_0_add_3.0.1619492699305998.npy                                        |
│  |- TxtFile: ./dump/cpu/InceptionV3_InceptionV3_Mixed_7a_Branch_0_add_3.0.1619492699305998.npy.txt                                    |
│  |- NpySpec: [Shape: (32, 8, 8, 320)] [Dtype: float32] [Max: 5.8425903] [Min: -8.374472] [Mean: -0.7256237]                           │
│ NumCnt:   655360                                                                                                                      │
│ AllClose: False                                                                                                                       │
│ CosSim:   0.99999493                                                                                                                  │
│ ErrorPer: 0.023504638671875  (rl= 0.005, al= 0.002)                                                                                   │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

