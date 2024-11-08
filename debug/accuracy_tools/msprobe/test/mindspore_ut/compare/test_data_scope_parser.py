import unittest
from msprobe.core.compare.data_scope_parser import *
from msprobe.core.common.const import Const


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

        self.frame_func_lines = [
            "File /home/user/envs/python3.9/site-packages/mindspore/common/tensor.py, line 2465, \
            in copy, x = x / 1.0",
            "File /home/user/envs/python3.9/site-packages/mindformers/tensor_parallel/layers.py, line 1147, \
            in construct, \n masked_input = input_.copy() = self.vocab_start_index",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 2455, \
            in _backward_hook_construct, \n outputs = self.construct(outputs, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 494, \
            in _run_construct, \n output = self._backward_hook_construct(*inputs, **kwargs)",
            "File /home/user/envs/python3.9/site-packages/mindspore/nn/cell.py, line 733, \
            in __call__, \n return self._run_construct(*args, *kwargs)"
        ]

        self.simplified_lines = [
            "File /home/user/envs/python3.9/site-packages/mindspore/common/tensor.py, line 2465, \
            in copy, \n x = x / 1.0",
            "File /home/user/envs/python3.9/site-packages/mindformers/tensor_parallel/layers.py, line 1147, \
            in construct, \n masked_input = input_.copy() = self.vocab_start_index"
        ]

        self.pt_construct = {
            "Functional.max_pool2d.0.forward": "Module.pool1.MaxPool2d.forward.0",
            "Functional.conv2d.1.forward": "Module.conv2.Conv2d.forward.0",
            "Functional.linear.5.backward": "Module.fc3.Linear.backward.1",
            "Module.conv1.Conv2d.backward.1": None,
            "Functional.conv2d.2.forward": None
        }

        self.ms_construct = {
            "Functional.add.0.forward": "Cell.transformer_layers.0.attention.core_attention.scale_mask_softmax.ScaleMaskSoftmax.forward.0",
            "Tensor.reshape.2.forward": "Cell.transformer_layers.0.attention.ParallelAttention.forward.0",
            "Functional.add.4.forward": "Cell.transformer_layers.0.attention.core_attention.scale_mask_softmax.ScaleMaskSoftmax.forward.0"
        }

        self.ms_dump = {
            "data": {"Functional.add.0.forward": "",
                     "Functional.add.4.forward": "",
                     "Tensor.reshape.2.forward": ""
                     }
        }
        self.pt_dump = {
            "data": {
                "Functional.max_pool2d.0.forward": "",
                "Functional.conv2d.1.forward": "",
                "Functional.linear.5.backward": "",
                "Functional.conv2d.2.forward": "",
                "Module.conv1.Conv2d.backward.1": ""
            }
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
                "File /home/user/envs/lib/python3.9/site-packages/msprobe/pytorch/hook_module/wrap_Functional.py, line 97, \
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
            "Functional.conv2d.1.forward": [
                "File /home/user/envs/lib/python3.9/site-packages/msprobe/pytorch/hook_module/wrap_Functional.py, line 97, \
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
        start_pos, end_pos = find_regard_scope(
            self.lines, start_sign, end_sign)
        self.assertEqual(start_pos, 4)
        self.assertEqual(end_pos, 9)

    def test_find_stack_func_list(self):
        result = find_stack_func_list(self.lines)
        self.assertEqual(result, ([], ['attn_mask_add']))

    def test_get_dump_data_items_when_pt_valid_then_pass(self):
        result = get_dump_data_items(
            self.pt_dump, self.pt_stack, self.pt_construct, Const.PT_FRAMEWORK)
        expected_result = [
            {
                "data_name": "Functional.max_pool2d.0.forward",
                "construct_scope": 'Module.pool1.MaxPool2d.forward.0',
                "full_scope": "Module.pool1.max_pool2d"
            },
            {
                "data_name": "Functional.conv2d.1.forward",
                "construct_scope": "Module.conv2.Conv2d.forward.0",
                "full_scope": "Module.conv2.conv2d"
            },
            {
                "data_name": "Functional.linear.5.backward",
                "construct_scope": "Module.fc3.Linear.backward.1",
                "full_scope": "Module.fc3.linear"
            },
            {
                "data_name": "Functional.conv2d.2.forward",
                "construct_scope": None,
                "full_scope": "Module.conv2d"
            },
            {
                "data_name": "Module.conv1.Conv2d.backward.1",
                "construct_scope": None,
                "full_scope": "Module.conv1"
            }
        ]
        actual_values = [(res.data_name, res.construct_scope) for res in result]
        expect_values = [(item.get("data_name"), item.get("construct_scope")) for item in expected_result]
        self.assertListEqual(actual_values, expect_values)

    def test_get_dump_data_items_when_ms_valid_then_pass(self):
        result = get_dump_data_items(
            self.ms_dump, self.ms_stack, self.ms_construct, Const.MS_FRAMEWORK)
        expected_result = [
            {
                "data_name": "Functional.add.0.forward",
                "construct_scope": "Cell.transformer_layers.0.attention.core_attention.scale_mask_softmax.ScaleMaskSoftmax.forward.0",
                "full_scope": "Cell.transformer_layers.0.attention.core_attention.scale_mask_softmax.attn_mask_add.add"
            },
            {
                "data_name": "Functional.add.4.forward",
                "construct_scope": "Cell.transformer_layers.0.attention.core_attention.scale_mask_softmax.ScaleMaskSoftmax.forward.0",
                "full_scope": "Cell.transformer_layers.0.attention.core_attention.scale_mask_softmax.attn_mask_add.add"
            },
            {
                "data_name": "Tensor.reshape.2.forward",
                "construct_scope": "Cell.transformer_layers.0.attention.ParallelAttention.forward.0",
                "full_scope": "Cell.transformer_layers.0.attention.reshape"
            }
        ]
        actual_values = [(res.data_name, res.construct_scope) for res in result]
        expect_values = [(item.get("data_name"), item.get("construct_scope")) for item in expected_result]
        self.assertListEqual(actual_values, expect_values)

    def test_get_stack_in_lines_when_frame_func_then_pass(self):
        result = get_stack_in_lines(self.simplified_lines)
        expected_result = ["copy"]
        self.assertEqual(result, expected_result)

    def test_find_stack_func_list_when_frame_func_then_pass(self):
        result = find_stack_func_list(self.frame_func_lines)
        self.assertEqual(result, (["copy"], []))
