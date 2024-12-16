# test_code_mapping.py
import unittest
import tempfile
import os
import re
import argparse
import numpy as np
from unittest.mock import patch
from pathlib import Path

# 从你的项目中导入相关函数和类
from msprobe.mindspore.code_mapping.cmd_parser import add_ir_parser_arguments
from msprobe.mindspore.code_mapping.main import code_mapping_main

# 将ir和csv的测试内容提取为独立变量
TEST_IR_CONTENT = """# IR entry: @19_1___main___Net_construct_72
# Total subgraphs: 3

# Attrs:
has_shard: 0
flash_sp_send_recv_has_attached: 1
has_attached: 1
check_set_strategy_valid_once_only: 1
FLASH_SP_RUN_ONCE_ONLY: 1
FIAS_SP_RUN_ONCE_ONLY: 1
less_bn: 0
auto_parallel_finish_pre_action: 1

# Total params: 2
# Params:
%para1_x: <Tensor[Float32], ()> : []
%para2_y: <Tensor[Float32], ()> : []

Node counting information:
Total number of nodes: 29
Total number of cnodes: 12

subgraph attr:
has_shard: 0
flash_sp_send_recv_has_attached: 1
has_attached: 1
check_set_strategy_valid_once_only: 1
FLASH_SP_RUN_ONCE_ONLY: 1
FIAS_SP_RUN_ONCE_ONLY: 1
less_bn: 0
auto_parallel_finish_pre_action: 1
subgraph instance: 19_1___main___Net_construct_72 : 0xaaaae0ca0250
# In file test_ir.py:15~20, 4~16/    def construct(self, x, y):/
subgraph @19_1___main___Net_construct_72() {
  %0(CNode_69$a) = PrimFunc_Sub(%para1_x, Tensor(shape=[], dtype=Float32, value=1)) cnode_attrs: {checkpoint: Bool(1)}
      : (<Tensor[Float32], ()>, <Tensor[Float32], (), value=...>) -> (<Tensor[Float32], ()>)
      # Fullname with scope: (Default/Sub-op0)
      # In file test_ir.py:15~20, 4~16/    def construct(self, x, y):/
      # In file test_ir.py:16, 12~25/        a = ops.sub(x, 1)/
      # In file test_ir.py:16, 12~19/        a = ops.sub(x, 1)/<~~This line of code can be shared by multiple nodes, and may be duplicated./
      # In file /home/maoyanlongbak/anaconda3/envs/pytorch21copy/lib/python3.8/site-packages/mindspore/ops/auto_generate/gen_ops_def.py:7431~7474, 0~31/        >>> y = Tensor(1, mindspore.int32)/
      # In file /home/maoyanlongbak/anaconda3/envs/pytorch21copy/lib/python3.8/site-packages/mindspore/ops/auto_generate/gen_ops_def.py:7474, 11~31/    Supported Platforms:/
  %1(CNode_70$b) = PrimFunc_Add(%0, %para2_y) cnode_attrs: {checkpoint: Bool(1)}
      : (<Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<Tensor[Float32], ()>)
      # Fullname with scope: (Default/Add-op0)
      # In file test_ir.py:15~20, 4~16/    def construct(self, x, y):/
      # In file test_ir.py:17, 12~25/        b = ops.add(a, y)/
      # In file test_ir.py:17, 12~19/        b = ops.add(a, y)/<~~This line of code can be shared by multiple nodes, and may be duplicated./
      # In file /home/maoyanlongbak/anaconda3/envs/pytorch21copy/lib/python3.8/site-packages/mindspore/ops/auto_generate/gen_ops_def.py:312~370, 0~31/def add(input, other):/
      # In file /home/maoyanlongbak/anaconda3/envs/pytorch21copy/lib/python3.8/site-packages/mindspore/ops/auto_generate/gen_ops_def.py:370, 11~31/    return add_op(input, other)/
  %2(CNode_71) = PrimFunc_Cast(%1, I64(30)) primitive_attrs: {output_names: [output], input_names: [x, dst_type]} cnode_attrs: {checkpoint: Bool(1)}
      : (<Tensor[Float32], ()>, <Int64, NoShape>) -> (<Tensor[Bool], ()>)
      # Fullname with scope: (Default/Cast-op0)
      # In file /home/maoyanlongbak/anaconda3/envs/pytorch21copy/lib/python3.8/site-packages/mindspore/_extends/parse/standard_method.py:2755~2757, 0~23/def bool_(x):/
      # In file /home/maoyanlongbak/anaconda3/envs/pytorch21copy/lib/python3.8/site-packages/mindspore/_extends/parse/standard_method.py:2757, 11~23/    return x.__bool__()/
      # In file /home/maoyanlongbak/anaconda3/envs/pytorch21copy/lib/python3.8/site-packages/mindspore/_extends/parse/standard_method.py:2757, 11~21/    return x.__bool__()/
      # In file /home/maoyanlongbak/anaconda3/envs/pytorch21copy/lib/python3.8/site-packages/mindspore/_extends/parse/standard_method.py:3275~3280, 0~34/def tensor_bool(x):/
      # In file /home/maoyanlongbak/anaconda3/envs/pytorch21copy/lib/python3.8/site-packages/mindspore/_extends/parse/standard_method.py:3278~3279, 4~38/    if is_cond and F.isconstant(x):/
      # In file /home/maoyanlongbak/anaconda3/envs/pytorch21copy/lib/python3.8/site-packages/mindspore/_extends/parse/standard_method.py:3280, 11~34/    return F.cast(x, mstype.bool_)/
  %3(CNode_80) = Partial(@20_4_✓__main___Net_construct_75, %1, %0) primitive_attrs: {side_effect_propagate: I64(1)} cnode_attrs: {checkpoint: Bool(1)}
      : (<Func, NoShape>, <Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<Func, NoShape>)
      # Fullname with scope: (Default/Partial-op0)
  %4(CNode_82) = Partial(@21_14_✗__main___Net_construct_76, %1) primitive_attrs: {side_effect_propagate: I64(1)} cnode_attrs: {checkpoint: Bool(1)}
      : (<Func, NoShape>, <Tensor[Float32], ()>) -> (<Func, NoShape>)
      # Fullname with scope: (Default/Partial-op1)
  %5(CNode_74) = Switch(%2, %3, %4) cnode_attrs: {checkpoint: Bool(1)}
      : (<Tensor[Bool], ()>, <Func, NoShape>, <Func, NoShape>) -> (<Func, NoShape>)
      # Fullname with scope: (Default/Switch-op0)
      # In file test_ir.py:15~20, 4~16/    def construct(self, x, y):/
      # In file test_ir.py:18~19, 8~43/        if b :/
  %6(CNode_77) = %5[@FuncUnion(@20_4_✓__main___Net_construct_75, @21_14_✗__main___Net_construct_76)]()
      : () -> (<Tensor[Float32], ()>)
      # Fullname with scope: (0)
      # In file test_ir.py:15~20, 4~16/    def construct(self, x, y):/
      # In file test_ir.py:18~19, 8~43/        if b :/
  Return(%6) cnode_attrs: {checkpoint: Bool(1)}
      : (<Tensor[Float32], ()>)
      # Fullname with scope: (Default/Return-op2)
      # In file test_ir.py:15~20, 4~16/    def construct(self, x, y):/
      # In file test_ir.py:18~19, 8~43/        if b :/
}


indirect: 1
subgraph attr:
defer_inline: 0
undeterminate: 0
subgraph instance: 20_4_✓__main___Net_construct_75 : 0xaaaae0c9dc10
# Parameters: 2, (<Tensor[Float32], ()>, <Tensor[Float32], ()>)
# In file test_ir.py:15~20, 4~16/    def construct(self, x, y):/
subgraph @20_4_✓__main___Net_construct_75(%para3_Parameter_79, %para4_Parameter_78) {
  %0(output) = PrimFunc_Div(%para4_Parameter_78, %para3_Parameter_79)
      : (<Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<Tensor[Float32], ()>)
      # Fullname with scope: (Default/Div-op0)
      # In file test_ir.py:15~20, 4~16/    def construct(self, x, y):/
      # In file test_ir.py:19, 27~42/            b = ops.mul(b, self.func(a, b))/
      # In file test_ir.py:19, 27~36/            b = ops.mul(b, self.func(a, b))/<~~This line of code can be shared by multiple nodes, and may be duplicated./
      # In file test_ir.py:12~13, 4~28/    def func(x, y):/
      # In file test_ir.py:13, 15~28/        return ops.div(x, y)/
      # In file test_ir.py:13, 15~22/        return ops.div(x, y)/<~~This line of code can be shared by multiple nodes, and may be duplicated./
      # In file /home/maoyanlongbak/anaconda3/envs/pytorch21copy/lib/python3.8/site-packages/mindspore/ops/function/math_func.py:727~786, 0~17/def div(input, other, *, rounding_mode=None):/
      # In file /home/maoyanlongbak/anaconda3/envs/pytorch21copy/lib/python3.8/site-packages/mindspore/ops/function/math_func.py:782~785, 4~42/    if rounding_mode:/
      # In file /home/maoyanlongbak/anaconda3/envs/pytorch21copy/lib/python3.8/site-packages/mindspore/ops/function/math_func.py:785, 17~42/        output = tensor_div_(input, other)/<~~This line of code can be shared by multiple nodes, and may be duplicated./
  %1(CNode_73$b) = PrimFunc_Mul(%para3_Parameter_79, %0)
      : (<Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<Tensor[Float32], ()>)
      # Fullname with scope: (Default/Mul-op0)
      # In file test_ir.py:15~20, 4~16/    def construct(self, x, y):/
      # In file test_ir.py:19, 16~43/            b = ops.mul(b, self.func(a, b))/
      # In file test_ir.py:19, 16~23/            b = ops.mul(b, self.func(a, b))/<~~This line of code can be shared by multiple nodes, and may be duplicated./
      # In file /home/maoyanlongbak/anaconda3/envs/pytorch21copy/lib/python3.8/site-packages/mindspore/ops/auto_generate/gen_ops_def.py:5222~5269, 0~31/        >>> import mindspore/
      # In file /home/maoyanlongbak/anaconda3/envs/pytorch21copy/lib/python3.8/site-packages/mindspore/ops/auto_generate/gen_ops_def.py:5269, 11~31/    Supported Platforms:/
  Return(%1)
      : (<Tensor[Float32], ()>)
      # Fullname with scope: (Default/Return-op0)
      # In file test_ir.py:15~20, 4~16/    def construct(self, x, y):/
      # In file test_ir.py:19, 12~43/            b = ops.mul(b, self.func(a, b))/
}


indirect: 1
subgraph attr:
defer_inline: 0
undeterminate: 0
subgraph instance: 21_14_✗__main___Net_construct_76 : 0xaaaae0c71c00
# Parameters: 1, (<Tensor[Float32], ()>)
# In file test_ir.py:15~20, 4~16/    def construct(self, x, y):/
subgraph @21_14_✗__main___Net_construct_76(%para5_Parameter_81) {
  Return(%para5_Parameter_81)
      : (<Tensor[Float32], ()>)
      # Fullname with scope: (Default/Return-op1)
      # In file test_ir.py:15~20, 4~16/    def construct(self, x, y):/
      # In file test_ir.py:18~19, 8~43/        if b :/
}"""

TEST_CSV_CONTENT = """Op Type,Op Name,Task ID,Stream ID,Timestamp,IO,Slot,Data Size,Data Type,Shape,Max Value,Min Value,L2Norm Value
Sub,Default_Sub-op0,0,0,1733905446819790,input,0,4,float32,"()",3,3,3,
Sub,Default_Sub-op0,0,0,1733905446820357,input,1,4,float32,"()",1,1,1,
Sub,Default_Sub-op0,0,0,1733905446820495,output,0,4,float32,"()",2,2,2,
Add,Default_Add-op0,0,0,1733905446822806,input,0,4,float32,"()",2,2,2,
Add,Default_Add-op0,0,0,1733905446822996,input,1,4,float32,"()",2,2,2,
Add,Default_Add-op0,0,0,1733905446823151,output,0,4,float32,"()",4,4,4,
Cast,Default_Cast-op0,0,0,1733905446823900,input,0,4,float32,"()",4,4,4,
Cast,Default_Cast-op0,0,0,1733905446824053,input,1,8,int64,"()",30,30,30,
Cast,Default_Cast-op0,0,0,1733905446824184,output,0,1,bool,"()",1,1,1,
Div,Default_Div-op0,0,0,1733905446827858,input,0,4,float32,"()",2,2,2,
Div,Default_Div-op0,0,0,1733905446828193,input,1,4,float32,"()",4,4,4,
Div,Default_Div-op0,0,0,1733905446828341,output,0,4,float32,"()",0.5,0.5,0.5,
Mul,Default_Mul-op0,0,0,1733905446831139,input,0,4,float32,"()",4,4,4,
Mul,Default_Mul-op0,0,0,1733905446831365,input,1,4,float32,"()",0.5,0.5,0.5,
Mul,Default_Mul-op0,0,0,1733905446831510,output,0,4,float32,"()",2,2,2,
"""


class TestCodeMapping(unittest.TestCase):
    def test_statistic_code_mapping(self):
        # 使用临时目录创建并测试
        with tempfile.TemporaryDirectory() as tmpdir:
            ir_file_path = os.path.join(tmpdir, "18_validate_0068.ir")
            csv_file_path = os.path.join(tmpdir, "statistic.csv")

            # 创建ir文件
            with open(ir_file_path, 'w') as f:
                f.write(TEST_IR_CONTENT)

            # 创建csv文件
            with open(csv_file_path, 'w') as f:
                f.write(TEST_CSV_CONTENT)

            # 读取运行前的CSV文件内容
            with open(csv_file_path, 'r') as f:
                original_csv_content = f.read()

            # 准备参数
            parser = argparse.ArgumentParser()
            add_ir_parser_arguments(parser)

            args = parser.parse_args(["--ir", ir_file_path, "--dump_data", csv_file_path, "--output", tmpdir])

            # 执行主函数
            code_mapping_main(args)

            # 读取运行后的CSV文件内容
            with open(csv_file_path, 'r') as f:
                updated_csv_content = f.read()

            # 比较文件内容是否有变化
            self.assertNotEqual(original_csv_content, updated_csv_content, "CSV文件内容未变化，测试失败")

    def test_npy_code_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ir_file_path = os.path.join(tmpdir, "18_validate_0068.ir")

            # 创建IR文件
            with open(ir_file_path, 'w') as f:
                f.write(TEST_IR_CONTENT)

            # 创建包含npy文件的data目录
            data_dir = os.path.join(tmpdir, "data_dir")
            os.makedirs(data_dir, exist_ok=True)

            # 准备需要创建的npy文件名列表
            npy_files = ["Add.Default_Add-op0.0.0.1734008383669918.input.0.DefaultFormat.float32.npy"]

            # 创建空的npy文件，或写入一些数据
            dummy_data = np.array([1.0, 2.0, 3.0])
            for fname in npy_files:
                file_path = os.path.join(data_dir, fname)
                np.save(file_path, dummy_data)  # 写入测试数据

            # 准备参数
            parser = argparse.ArgumentParser()
            add_ir_parser_arguments(parser)

            # 此处 --dump_data 参数传入我们创建的data目录
            args = parser.parse_args(["--ir", ir_file_path, "--dump_data", data_dir, "--output", tmpdir])

            # 执行主函数
            code_mapping_main(args)

            # 使用正则表达式验证生成的文件名
            pattern = r"code_mapping_\d{8}\d{6}\.csv"  # 匹配如 code_mapping_YYYYMMDDHHMMSS.csv 格式
            generated_files = os.listdir(tmpdir)

            # 检查是否有符合格式的文件生成
            matching_files = [f for f in generated_files if re.match(pattern, f)]
            self.assertTrue(matching_files,
                            msg=f"没有生成符合要求的CSV文件。生成的文件列表: {generated_files}. 正则匹配模式: {pattern}")


if __name__ == '__main__':
    unittest.main()
