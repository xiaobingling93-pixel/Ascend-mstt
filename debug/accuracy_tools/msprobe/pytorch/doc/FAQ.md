# 精度预检工具

1. 预检工具在dump和run_ut的过程中，是否需要同时开启或关闭jit编译（jit_compile）？

   答：是。

2. 预检工具对于type_as这类涉及数据类型转换操作的API，是否具有参考性？

   由于这类API在CPU侧存在精度先提升后下降的操作，因此这类API的有效性的参考价值有限。

3. run ut过程中出现报错：ERROR:Got unsupported ScalarType BFloat16

   答：请使用最新版本的工具。

4. Dropout算子，CPU和NPU的随机应该不一样，为什么结果比对是一致的？

   答：这个结果是正常的，工具对该算子有特殊处理，只判定位置为0的位置比例大约和设定p值相当。

5. 为什么浮点型数据bench和CPU的dtype不一致？

   答：对于fp16的数据，CPU会上升一个精度fp32去计算，这是和算子那边对齐的精度结论，CPU用更高精度去计算会更接近真实值。

6. 添加预检工具后截取操作报错：`IndexError: too many indices for tensor of dimension x` 或 `TypeError: len() of a 0-d tensor`。

   答：注释工具目录mstt/debug/accuracy_tools/atat/pytorch/hook_module/support_wrap_ops.yaml文件中Tensor:下的`- __getitem__`，工具会跳过dump该API。如果是需要dump的关键位置API也可以考虑根据报错堆栈信息注释引发报错的类型检查。

7. 添加预检工具后F.gelu触发ValueError报错：`activation_func must be F.gelu`等。

   答：注释工具目录mstt/debug/accuracy_tools/atat/pytorch/hook_module/support_wrap_ops.yaml文件中functional:下的的`- gelu`，工具会跳过dump该API。如果是需要dump的关键位置API也可以考虑根据报错堆栈信息注释引发报错的类型检查。

8. 添加预检工具后触发AsStrided算子相关的报错，或者编译相关的报错，如：`Failed to compile Op [AsStrided]`。

   答：注释工具目录mstt/debug/accuracy_tools/atat/pytorch/hook_module/support_wrap_ops.yaml文件中Tensor:下的`- t`和`- transpose`。

9. Tensor 魔法函数具体对应什么操作？

   答：

   | Tensor魔法函数  | 具体操作         |
   | --------------- | ---------------- |
   | `__add__`       | +                |
   | `__and__`       | &                |
   | `__bool__`      | 返回Tensor布尔值 |
   | `__div__`       | /                |
   | `__eq__`        | ==               |
   | `__ge__`        | >=               |
   | `__gt__`        | >                |
   | `__iadd__`      | +=               |
   | `__iand__`      | &=               |
   | `__idiv__`      | /=               |
   | `__ifloordiv__` | //=              |
   | `__ilshift__`   | <<=              |
   | `__imod__`      | %=               |
   | `__imul__`      | *=               |
   | `__ior__`       | \|=              |
   | `__irshift__`   | >>=              |
   | `__isub__`      | -=               |
   | `__ixor__`      | ^=               |
   | `__lshift__`    | <<               |
   | `__matmul__`    | 矩阵乘法         |
   | `__mod__`       | %                |
   | `__mul__`       | *                |
   | `__nonzero__`   | 同`__bool__`     |
   | `__or__`        | \|               |
   | `__radd__`      | +（反向）        |
   | `__rmul__`      | *（反向）        |
   | `__rshift__`    | >>               |
   | `__sub__`       | -                |
   | `__truediv__`   | 同`__div__`      |
   | `__xor__`       | ^                |

# 精度比对工具

## 工具使用

### dump指定融合算子

dump指定操作当前支持dump指定融合算子的输入输出，需要在mstt/debug/accuracy_tools/atat/pytorch/hook_module/support_wrap_ops.yaml中添加，比如以下代码段调用的softmax融合算子

```
def npu_forward_fused_softmax(self, input_, mask):
    resl = torch_npu.npu_scaled_masked_softmax(input_, mask, self.scale, False)
    return resl
```

如果需要dump其中调用的npu_scaled_masked_softmax算子的输入输出信息，需要在support_wrap_ops.yaml中的torch_npu: 中自行添加该融合算子即可：

```
- npu_scaled_masked_softmax
```

（npu_scaled_masked_softmax融合算子工具已支持dump，本例仅供参考）

## 常见问题

### 1. 在同一个目录多次执行dump会冲突吗？

会，同一个目录多次dump，会覆盖上一次结果，可以使用dump_path参数修改dump目录。

### 2. 如何dump算子级的数据？

需要配置level为L2模式。

### 3. 工具比对发现NPU和标杆数据的API无法完全对齐？

torch版本和硬件差异属于正常情况。

## 异常情况

### 2. HCCL 报错： error code: EI0006

**故障现象**

使用atat工具时，报错： error code: EI0006。

**故障原因**

CANN软件版本较低导致不兼容。

**故障处理**

升级新版CANN软件版本。

### 3. torch_npu._C._clear_overflow_npu() RuntimeError NPU error，error code is 107002

如果运行溢出检测功能遇到这个报错，采取以下解决方法：
如果是单卡运行，添加如下代码，0是卡号，选择自己空闲的卡号。

```
torch.npu.set_device('npu:0')
```

如果多卡运行，请在代码中修改对应卡号，比如进程使用卡号为{rank}时可以添加如下代码：

```
torch.npu.set_device(f'npu:{rank}')
```

如果运行精度比对功能遇到这个报错，尝试安装最新版本的atat。

### 4. dump得到的VF_lstm_99_forward_input.1.0.npy、VF_lstm_99_forward_input.1.1.npy类似的数据是否正常？

带1.0/1.1/1.2后缀的npy是正常现象，例如当输入数据为[[tensor1, tensor2, tensor3]]会生成这样的后缀。

### 5. 进行compare报错：The current file contains stack information, please turn on the stack_mode

在比对脚本中，设置stack_mode=True，例如：

```
from atat.pytorch import compare
dump_result_param={
"npu_json_path": "./npu_dump/dump.json",
"bench_json_path": "./gpu_dump/dump.json",
"stack_json_path": "./npu_dump/stack.json",
"is_print_compare_log": True
}
compare(dump_result_param, output_path="./output", stack_mode=True)
```

### 6. dump指定反向API的kernel级别的数据报错：NameError：name 'torch_npu' is not defined

- 如果是npu环境，请安装torch_npu；
- 如果是gpu环境，暂不支持dump指定API的kernel级别的数据

### 7. 配置dump_path后，使用工具报错：[ERROR]The file path /home/xxx/dump contains special characters

- 请检查你设置的dump绝对路径是否包含特殊字符，确保路径名只包含大小写字母、数字、下划线、斜杠、点和短横线
- 注意，如果执行脚本的路径为/home/abc++/，设置的dump_path="./dump"，工具实际校验的路径为绝对路径/home/abc++/dump，++为特殊字符，会引发本条报错

### 8. 无法dump matmul权重的反向梯度数据

- matmul期望的输入是二维，当输入不是二维时，会将输入通过view操作展成二维，再进行matmul运算，因此在反向求导时，backward_hook能拿到的是UnsafeViewBackward这步操作里面数据的梯度信息，取不到MmBackward这步操作里面数据的梯度信息，即权重的反向梯度数据。
- 典型的例子有，当linear的输入不是二维，且无bias时，会调用output = input.matmul(weight.t()),因此拿不到linear层的weight的反向梯度数据。

### 9. dump.json文件中的某些api的dtype类型为float16，但是读取此api的npy文件显示的dtype类型为float32

- atat工具在dump数据时需要将原始数据从npu to cpu上再转换为numpy类型，npu to cpu的逻辑和gpu to cpu是保持一致的，都存在dtype可能从float16变为float32类型的情况，如果出现dtype不一致的问题，最终dump数据的dtype以pkl文件为准。

### 10. 使用dataloader后raise异常Exception("atat: exit after iteration {}". format(max(self.config.step))

- 正常现象，dataloader通过raise结束程序，堆栈信息可忽略。

### 11. 添加atat工具后截取操作报错：`IndexError: too many indices for tensor of dimension x` 或 `TypeError: len() of a 0-d tensor`。

- 注释工具目录mstt/debug/accuracy_tools/atat/pytorch/hook_module/support_wrap_ops.yaml文件中Tensor:下的`- __getitem__`，工具会跳过dump该API。如果是需要dump的关键位置API也可以考虑根据报错堆栈信息注释引发报错的类型检查。

### 12. 添加atat工具后F.gelu触发ValueError报错：`activation_func must be F.gelu`等。

- 注释工具目录mstt/debug/accuracy_tools/atat/pytorch/hook_module/support_wrap_ops.yaml文件中functional:下的的`- gelu`，工具会跳过dump该API。如果是需要dump的关键位置api也可以考虑根据报错堆栈信息注释引发报错的类型检查。

### 13. 添加atat工具后触发AsStrided算子相关的报错，或者编译相关的报错，如：`Failed to compile Op [AsStrided]`。

- 注释工具目录mstt/debug/accuracy_tools/atat/pytorch/hook_module/support_wrap_ops.yaml文件中Tensor:下的`- t`和`- transpose`。
