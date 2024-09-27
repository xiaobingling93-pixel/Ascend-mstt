import unittest
from msprobe.mindspore.compare.modify_mapping import *
from unittest.mock import patch, MagicMock

class TestModifyMapping(unittest.TestCase):

    def setUp(self):
        # 这里可以设置常量和初始化
        self.lines = [
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 507, \
            in _run_construct, \n output = self._run_forward_hook(inputs, output)", 
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 745, \
            in __call__, \n return self._run_construct(*args, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/msprobe/mindspore/dump/hook_cell/hook_cell.py, line 48, \
            in __call__, \n out = super(HOOKCell, self).__call__(*args, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/msprobe/mindspore/dump/hook_cell/wrap_api.py, line 92, \
            in api_function, \n return ApiTemplate(api_name, api_dict, prefix, hook)(*args, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/mindformers/transformer/utils.py, line 38, \
            in attn_mask_add, \n attention_scores = ops.add(",
            "File /home/user/envs/python3.9/site-packages/mindformers/transformer/scale_mask_softmax.py, line 65, \
            in construct, \n masked_input = self.mask_func(x, mask) if mask is not None else x",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2460, \
            in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 498, \
            in _run_construct, \n output = self._backward_hook_construct(*outputs, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 745, \
            in __call__, \n return self._run_construct(*args, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/mindformers/transformer/transformer.py, line 533, \
            in construct, \n attention_output = self.attention(norm_output, attention_mask, rotary_pos_emb)",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2462, \
            in _backward_hook_construct, \n outputs = self.construct(output, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2460, \
            in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 498, \
            in _run_construct, \n output = self._backward_hook_construct(*outputs, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 745, \
            in __call__, \n return self._run_construct(*args, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/mindformers/transformer/transformer.py, line 533, \
            in construct, \n attention_output = self.attention(norm_output, attention_mask, rotary_pos_emb)",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2462, \
            in _backward_hook_construct, \n outputs = self.construct(output, **kwargs)"
        ]
        self.pt_construct = {
            "Functional.max_pool2d.0.forward": "Module.pool1.MaxPool2d.forward.0",
            "Funtional.conv2d.1.forward": "Module.conv2.Conv2d.forward.0",
            "Functional.linear.5.backward": "Module.fc3.Linear.backward.1",
            "Module.conv1.Conv2d.backward.1": None
        }
        self.ms_construct = {
            "Functional.add.0.forward": "Cell.transformer_layers.0.attention.core_attention.scale_mask_softmax.ScaleMaskSoftmax.forward.0",
            "Tensor.reshape.2.forward": "Cell.transformer_layers.0.attention.ParallelAttention.forward.0",
            "Functional.add.4.forward": "Cell.transformer_layers.0.attention.core_attention.scale_mask_softmax.ScaleMaskSoftmax.forward.0"
        }
        self.ms_stack = {
            "Functional.add.0.forward": [
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 507, \
                in _run_construct, \n output = self._run_forward_hook(inputs, output)", 
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 745, \
                in __call__, \n return self._run_construct(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/msprobe/mindspore/dump/hook_cell/hook_cell.py, line 48, \
                in __call__, \n out = super(HOOKCell, self).__call__(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/msprobe/mindspore/dump/hook_cell/wrap_api.py, line 92, \
                in api_function, \n return ApiTemplate(api_name, api_dict, prefix, hook)(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindformers/transformer/utils.py, line 38, \
                in attn_mask_add, \n attention_scores = ops.add(",
                "File /home/user/envs/python3.9/site-packages/mindformers/transformer/scale_mask_softmax.py, line 65, \
                in construct, \n masked_input = self.mask_func(x, mask) if mask is not None else x",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2460, \
                in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 498, \
                in _run_construct, \n output = self._backward_hook_construct(*outputs, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 745, \
                in __call__, \n return self._run_construct(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindformers/transformer/transformer.py, line 533, \
                in construct, \n attention_output = self.attention(norm_output, attention_mask, rotary_pos_emb)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2462, \
                in _backward_hook_construct, \n outputs = self.construct(output, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2460, \
                in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 498, \
                in _run_construct, \n output = self._backward_hook_construct(*outputs, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 745, \
                in __call__, \n return self._run_construct(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindformers/transformer/transformer.py, line 533, \
                in construct, \n attention_output = self.attention(norm_output, attention_mask, rotary_pos_emb)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2462, \
                in _backward_hook_construct, \n outputs = self.construct(output, **kwargs)"
            ],
            "Tensor.reshape.2.forward": [
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 507, \
                in _run_construct, \n output = self._run_forward_hook(inputs, output)", 
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 745, \
                in __call__, \n return self._run_construct(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/msprobe/mindspore/dump/hook_cell/hook_cell.py, line 48, \
                in __call__, \n out = super(HOOKCell, self).__call__(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/msprobe/mindspore/dump/hook_cell/wrap_api.py, line 92, \
                in api_function, \n return ApiTemplate(api_name, api_dict, prefix, hook)(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindformers/transformer/transformer.py, line 367, \
                in construct, \n query = query.reshape(bs, seq_len, -1, self.head_dim).transpose((0, 2, 1, 3))",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2460, \
                in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 498, \
                in _run_construct, \n output = self._backward_hook_construct(*outputs, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 745, \
                in __call__, \n return self._run_construct(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindformers/transformer/transformer.py, line 533, \
                in construct, \n attention_output = self.attention(norm_output, attention_mask, rotary_pos_emb)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2462, \
                in _backward_hook_construct, \n outputs = self.construct(output, **kwargs)"
            ],
            "Functional.add.4.forward": [
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 507, \
                in _run_construct, \n output = self._run_forward_hook(inputs, output)", 
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 745, \
                in __call__, \n return self._run_construct(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/msprobe/mindspore/dump/hook_cell/hook_cell.py, line 48, \
                in __call__, \n out = super(HOOKCell, self).__call__(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/msprobe/mindspore/dump/hook_cell/wrap_api.py, line 92, \
                in api_function, \n return ApiTemplate(api_name, api_dict, prefix, hook)(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindformers/transformer/utils.py, line 38, \
                in attn_mask_add, \n attention_scores = ops.add(",
                "File /home/user/envs/python3.9/site-packages/mindformers/transformer/scale_mask_softmax.py, line 65, \
                in construct, \n masked_input = self.mask_func(x, mask) if mask is not None else x",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2460, \
                in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 498, \
                in _run_construct, \n output = self._backward_hook_construct(*outputs, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 745, \
                in __call__, \n return self._run_construct(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindformers/transformer/transformer.py, line 533, \
                in construct, \n attention_output = self.attention(norm_output, attention_mask, rotary_pos_emb)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2462, \
                in _backward_hook_construct, \n outputs = self.construct(output, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2460, \
                in _backward_hook_construct, \n outputs = self.construct(*outputs, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 498, \
                in _run_construct, \n output = self._backward_hook_construct(*outputs, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 745, \
                in __call__, \n return self._run_construct(*args, **kwargs)",
                "File /home/user/envs/python3.9/site-packages/mindformers/transformer/transformer.py, line 533, \
                in construct, \n attention_output = self.attention(norm_output, attention_mask, rotary_pos_emb)",
                "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2462, \
                in _backward_hook_construct, \n outputs = self.construct(output, **kwargs)"
            ]
        }
        self.pt_stack = {
            "Functional.max_pool2d.0.forward": [
                "File /home/user/envs/lib/python3.9/site-packages/msprobe/pytorch/hook_module/wrap_funtional.py, line 97, \
                in functional_op_template, \n return FunctionalOPTemplate(op_name, hook)(*args, **kwargs)",
                "File /home/user/envs/lib/python3.9/site-packages/torch/nn/modules/pooling.py, line 166, \
                in forward, \n return F.max_pool2d(input, self.kernel_size, self.stride,",
                "File /home/user/envs/lib/python3.9/site-packages/torch/nn/modules/module.py, line 1568, \
                in _call_impl, \n result = foward_call(*args, **kwargs)",
                "File /home/user/envs/lib/python3.9/site-packages/torch/nn/modules/module.py, line 1518, \
                in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
                "File lenet.py, line 28, in forward, \n x = self.pool1(x)",
                "File /home/user/envs/lib/python3.9/site-packages/torch/nn/modules/module.py, line 1527, \
                in _call_impl, \n return forward_call(*args, **kwargs)",
                "File /home/user/envs/lib/python3.9/site-packages/torch/nn/modules/module.py, line 1518, \
                in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
                "File lenet.py, line 47, in train_one_epoch, \n output = model(image)",
                "File lenet.py, line 83, in TestLeNetE2E, \n train_one_epoch(step, steps, train_loader, \
                model, optimizer, criterion)",
                "File lenet.py, line 88, in <module>, \n TestLeNetE2E()"
            ],
            "Funtional.conv2d.1.forward": [
                "File /home/user/envs/lib/python3.9/site-packages/msprobe/pytorch/hook_module/wrap_funtional.py, line 97, \
                in functional_op_template, \n return FunctionalOPTemplate(op_name, hook)(*args, **kwargs)",
                "File /home/user/envs/lib/python3.9/site-packages/torch/nn/modules/conv.py, line 456, \
                in _conv_forward, \n return F.conv2d(input, weight, bias, self.stride,",
                "File /home/user/envs/lib/python3.9/site-packages/torch/nn/modules/conv.py, line 460, \
                in forward, \n return self._conv_forward(input, self.weight, self.bias)",
                "File /home/user/envs/lib/python3.9/site-packages/torch/nn/modules/module.py, line 1568, \
                in _call_impl, \n result = foward_call(*args, **kwargs)",
                "File /home/user/envs/lib/python3.9/site-packages/torch/nn/modules/module.py, line 1518, \
                in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
                "File lenet.py, line 29, in forward, \n x = F.relu(self.conv2(x))",
                "File /home/user/envs/lib/python3.9/site-packages/torch/nn/modules/module.py, line 1527, \
                in _call_impl, \n return forward_call(*args, **kwargs)",
                "File /home/user/envs/lib/python3.9/site-packages/torch/nn/modules/module.py, line 1518, \
                in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)",
                "File lenet.py, line 47, in train_one_epoch, \n output = model(image)",
                "File lenet.py, line 83, in TestLeNetE2E, \n train_one_epoch(step, steps, train_loader, \
                model, optimizer, criterion)",
                "File lenet.py, line 88, in <module>, \n TestLeNetE2E()"
            ]
        }

    def test_find_regard_scope(self):
        start_sign = "add"
        end_sign = "attention"
        start_pos, end_pos = find_regard_scope(self.lines, start_sign, end_sign)
        self.assertEqual(start_pos, 4)
        self.assertEqual(end_pos, 9)

    def test_find_stack_func_list(self):
        result = find_stack_func_list(self.lines)
        self.assertEqual(result, ['attn_mask_add'])

    def test_get_duplicated_name(self):
        components = ["Module", "max_pool2d", "0", "forward"]
        duplicated_name = get_duplicated_name(components)
        self.assertEqual(duplicated_name, ["Module", "max_pool2d", "max_pool2d", "0", "forward"])

    def test_modify_mapping_with_stack_when_pt_valid_then_pass(self):
        result = modify_mapping_with_stack(self.pt_stack, self.pt_construct)
        expected_result = {
            "Module.pool1.max_pool2d.max_pool2d.0": {
                "origin_data": "Functional.max_pool2d.0.forward",
                "scope": "Module.pool1.MaxPool2d.forward.0",
                "stack": None
            },
            "Module.conv2.conv2d.conv2d.1": {
                "origin_data": "Module.conv1.Conv2d.backward.1",
                "scope": None,
                "stack": None
            },
            "Module.fc3.linear.linear.5": {
                "origin_data": "Functional.linear.5.backward",
                "scope": "Module.fc3.Linear.backward.1",
                "stack": None
            },
            "Module.conv1.Conv2d.1": {
                "origin_data": "Module.conv1.Conv2d.1",
                "scope": "Module.fc3.Linear.backward.1",
                "stack": None
            }
        }
        self.assertIn("Module.pool1.max_pool2d.max_pool2d.0", result)
        self.assertIn("Module.conv2.conv2d.conv2d.1", result)
        self.assertIn("Module.fc3.linear.linear.5", result)
        self.assertIn("Module.conv1.Conv2d.1", result)
        self.assertEqual(result["Module.pool1.max_pool2d.max_pool2d.0"]["origin_data"], "Functional.max_pool2d.0.forward")
        self.assertEqual(result["Module.conv1.Conv2d.1"]["origin_data"], "Module.conv1.Conv2d.backward.1")

    def test_modify_mapping_with_stack_when_ms_valid_then_pass(self):
        result = modify_mapping_with_stack(self.ms_stack, self.ms_construct)
        expected_result = {
            "Cell.transformer_layers.0.attention.core_attention.scale_mask_softmax.attn_mask_add.add.add.0": {
                "origin_data": "Functional.add.0.forward",
                "scope": "Cell.transformer_layers.0.attention.core_attention.\
                scale_mask_softmax.ScaleMaskSoftmax.forward.0",
                "stack": "attn_mask_add"
            },
            "Cell.transformer_layers.0.attention.reshape.reshape.2": {
                "origin_data": "Tensor.reshape.2.forward",
                "scope": "Cell.transformer_layers.0.attention.ParallelAttention.forward.0",
                "stack": None
            },
            "Cell.transformer_layers.0.attention.core_attention.scale_mask_softmax.attn_mask_add.add.add.4": {
                "origin_data": "Functional.add.4.forward",
                "scope": "Cell.transformer_layers.0.attention.core_attention.\
                scale_mask_softmax.ScaleMaskSoftmax.forward.0",
                "stack": "attn_mask_add"
            }
        }
        self.assertIn("Cell.transformer_layers.0.attention.core_attention.scale_mask_softmax.attn_mask_add.add.add.0", result)
        self.assertIn("Cell.transformer_layers.0.attention.reshape.reshape.2", result)
        self.assertIn("Cell.transformer_layers.0.attention.core_attention.scale_mask_softmax.attn_mask_add.add.add.4", result)
        self.assertEqual(result["Cell.transformer_layers.0.attention.core_attention.scale_mask_softmax.attn_mask_add.add.add.0"]["origin_data"], "Functional.add.0.forward")
        self.assertEqual(result["Cell.transformer_layers.0.attention.core_attention.scale_mask_softmax.attn_mask_add.add.add.0"]["scope"], "Cell.transformer_layers.0.attention.core_attention.scale_mask_softmax.ScaleMaskSoftmax.forward.0")
        self.assertEqual(result["Cell.transformer_layers.0.attention.core_attention.scale_mask_softmax.attn_mask_add.add.add.0"]["stack"], "attn_mask_add")
        self.assertEqual(result["Cell.transformer_layers.0.attention.reshape.reshape.2"]["origin_data"], "Tensor.reshape.2.forward")
        self.assertEqual(result["Cell.transformer_layers.0.attention.reshape.reshape.2"]["scope"], "Cell.transformer_layers.0.attention.ParallelAttention.forward.0")
