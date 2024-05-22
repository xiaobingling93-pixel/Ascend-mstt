# coding=utf-8
import unittest
import numpy as np
import os
from ptdbg_ascend.compare import acc_compare as compare
from ptdbg_ascend.common.utils import CompareConst


npu_dict = {'op_name': ['modulemodel.linear.Linear.0.forward_input.0', 'modulemodel.linear.Linear.0.forward_input.1', 'modulemodel.linear.Linear.0.forward_input.2.0', 'modulemodel.linear.Linear.0.forward_input.2.1', 'modulemodel.linear.Linear.0.forward_output.0'], 'input_struct': [('torch.float32', [10, 10], '7f84caad'), (None, None, None), ("<class 'int'>", '[]', None), ("<class 'int'>", '[]', None)], 'output_struct': [('torch.float32', [10, 10], '3e8354f5')], 'summery': [[2.8386683464050293, -2.158618688583374, 0.11464785784482956, 10.07983684539795], [None, None, None, None], [2, 2, 2, 2], [2, 2, 2, 2], [1.1663073301315308, -1.6045000553131104, -0.1430426388978958, 6.108779430389404]], 'stack_info': [['File run_sample.py, line 11, in forward, \n return self.relu(self.linear(x))', 'File /home/louyujing/miniconda3/envs/pytorch21/lib/python3.8/site-packages/torch/nn/modules/module.py, line 1568, in _call_impl, \n result = forward_call(*args, **kwargs)', 'File /home/louyujing/miniconda3/envs/pytorch21/lib/python3.8/site-packages/torch/nn/modules/module.py, line 1518, in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)', 'File run_sample.py, line 21, in forward, \n return self.linear(self.model(x))', 'File /home/louyujing/miniconda3/envs/pytorch21/lib/python3.8/site-packages/torch/nn/modules/module.py, line 1528, in _call_impl, \n return forward_call(*args, **kwargs)', 'File /home/louyujing/miniconda3/envs/pytorch21/lib/python3.8/site-packages/torch/nn/modules/module.py, line 1518, in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)', 'File run_sample.py, line 30, in , \n y = model(x)']]}
bench_dict = {'op_name': ['modulemodel.linear.Linear.0.forward_input.0', 'modulemodel.linear.Linear.0.forward_input.1', 'modulemodel.linear.Linear.0.forward_input.2.0', 'modulemodel.linear.Linear.0.forward_input.2.1', 'modulemodel.linear.Linear.0.forward_output.0'], 'input_struct': [('torch.float32', [10, 10], '7f84caad'), (None, None, None), ("<class 'int'>", '[]', None), ("<class 'int'>", '[]', None)], 'output_struct': [('torch.float32', [10, 10], '3e8354f5')], 'summery': [[2.8386683464050293, -2.158618688583374, 0.11464785784482956, 10.07983684539795], [None, None, None, None], [2, 2, 2, 2], [2, 2, 2, 2], [1.1663073301315308, -1.6045000553131104, -0.1430426388978958, 6.108779430389404]], 'stack_info': [['File run_sample.py, line 11, in forward, \n return self.relu(self.linear(x))', 'File /home/louyujing/miniconda3/envs/pytorch21/lib/python3.8/site-packages/torch/nn/modules/module.py, line 1568, in _call_impl, \n result = forward_call(*args, **kwargs)', 'File /home/louyujing/miniconda3/envs/pytorch21/lib/python3.8/site-packages/torch/nn/modules/module.py, line 1518, in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)', 'File run_sample.py, line 21, in forward, \n return self.linear(self.model(x))', 'File /home/louyujing/miniconda3/envs/pytorch21/lib/python3.8/site-packages/torch/nn/modules/module.py, line 1528, in _call_impl, \n return forward_call(*args, **kwargs)', 'File /home/louyujing/miniconda3/envs/pytorch21/lib/python3.8/site-packages/torch/nn/modules/module.py, line 1518, in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)', 'File run_sample.py, line 30, in , \n y = model(x)']]}
tensor_list = [{'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [10, 10], 'Max': 2.8386683464050293, 'Min': -2.158618688583374, 'Mean': 0.11464785784482956, 'Norm': 10.07983684539795, 'requires_grad': False, 'md5': '7f84caad', 'full_op_name': 'modulemodel.linear.Linear.0.forward_input.0'}, {'full_op_name': 'modulemodel.linear.Linear.0.forward_input.1', 'Max': None, 'Min': None, 'Mean': None, 'Norm': None, 'dtype': None, 'shape': None, 'md5': None, 'data_name': '-1'}, {'full_op_name': 'modulemodel.linear.Linear.0.forward_input.2.0', 'dtype': "<class 'int'>", 'shape': '[]', 'md5': None, 'Max': 2, 'Min': 2, 'Mean': 2, 'Norm': 2, 'data_name': '-1'}, {'full_op_name': 'modulemodel.linear.Linear.0.forward_input.2.1', 'dtype': "<class 'int'>", 'shape': '[]', 'md5': None, 'Max': 2, 'Min': 2, 'Mean': 2, 'Norm': 2, 'data_name': '-1'}, {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [10, 10], 'Max': 1.1663073301315308, 'Min': -1.6045000553131104, 'Mean': -0.1430426388978958, 'Norm': 6.108779430389404, 'requires_grad': True, 'md5': '3e8354f5', 'full_op_name': 'modulemodel.linear.Linear.0.forward_output.0'}, {'full_op_name': 'modulemodel.linear.Linear.0.forward', 'full_info': ['File run_sample.py, line 11, in forward, \n return self.relu(self.linear(x))', 'File /home/louyujing/miniconda3/envs/pytorch21/lib/python3.8/site-packages/torch/nn/modules/module.py, line 1568, in _call_impl, \n result = forward_call(*args, **kwargs)', 'File /home/louyujing/miniconda3/envs/pytorch21/lib/python3.8/site-packages/torch/nn/modules/module.py, line 1518, in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)', 'File run_sample.py, line 21, in forward, \n return self.linear(self.model(x))', 'File /home/louyujing/miniconda3/envs/pytorch21/lib/python3.8/site-packages/torch/nn/modules/module.py, line 1528, in _call_impl, \n return forward_call(*args, **kwargs)', 'File /home/louyujing/miniconda3/envs/pytorch21/lib/python3.8/site-packages/torch/nn/modules/module.py, line 1518, in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)', 'File run_sample.py, line 30, in , \n y = model(x)']}]
result_op_dict = {'op_name': ['modulemodel.linear.Linear.0.forward_input.0', 'modulemodel.linear.Linear.0.forward_input.1', 'modulemodel.linear.Linear.0.forward_input.2.0', 'modulemodel.linear.Linear.0.forward_input.2.1', 'modulemodel.linear.Linear.0.forward_output.0'], 'input_struct': [('torch.float32', [10, 10], '7f84caad'), (None, None, None), ("<class 'int'>", '[]', None), ("<class 'int'>", '[]', None)], 'output_struct': [('torch.float32', [10, 10], '3e8354f5')], 'summery': [[2.8386683464050293, -2.158618688583374, 0.11464785784482956, 10.07983684539795], [None, None, None, None], [2, 2, 2, 2], [2, 2, 2, 2], [1.1663073301315308, -1.6045000553131104, -0.1430426388978958, 6.108779430389404]], 'stack_info': [['File run_sample.py, line 11, in forward, \n return self.relu(self.linear(x))', 'File /home/louyujing/miniconda3/envs/pytorch21/lib/python3.8/site-packages/torch/nn/modules/module.py, line 1568, in _call_impl, \n result = forward_call(*args, **kwargs)', 'File /home/louyujing/miniconda3/envs/pytorch21/lib/python3.8/site-packages/torch/nn/modules/module.py, line 1518, in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)', 'File run_sample.py, line 21, in forward, \n return self.linear(self.model(x))', 'File /home/louyujing/miniconda3/envs/pytorch21/lib/python3.8/site-packages/torch/nn/modules/module.py, line 1528, in _call_impl, \n return forward_call(*args, **kwargs)', 'File /home/louyujing/miniconda3/envs/pytorch21/lib/python3.8/site-packages/torch/nn/modules/module.py, line 1518, in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)', 'File run_sample.py, line 30, in , \n y = model(x)']]}
o_result = [['modulemodel.linear.Linear.0.forward_input.0', 'modulemodel.linear.Linear.0.forward_input.0', 'torch.float32', 'torch.float32', [10, 10], [10, 10], '7f84caad', '7f84caad', 'Pass', ['File run_sample.py, line 11, in forward, \n return self.relu(self.linear(x))', 'File /home/louyujing/miniconda3/envs/pytorch21/lib/python3.8/site-packages/torch/nn/modules/module.py, line 1568, in _call_impl, \n result = forward_call(*args, **kwargs)', 'File /home/louyujing/miniconda3/envs/pytorch21/lib/python3.8/site-packages/torch/nn/modules/module.py, line 1518, in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)', 'File run_sample.py, line 21, in forward, \n return self.linear(self.model(x))', 'File /home/louyujing/miniconda3/envs/pytorch21/lib/python3.8/site-packages/torch/nn/modules/module.py, line 1528, in _call_impl, \n return forward_call(*args, **kwargs)', 'File /home/louyujing/miniconda3/envs/pytorch21/lib/python3.8/site-packages/torch/nn/modules/module.py, line 1518, in _wrapped_call_impl, \n return self._call_impl(*args, **kwargs)', 'File run_sample.py, line 30, in , \n y = model(x)']], ['modulemodel.linear.Linear.0.forward_input.1', 'modulemodel.linear.Linear.0.forward_input.1', None, None, None, None, None, None, 'Pass', 'None'], ['modulemodel.linear.Linear.0.forward_input.2.0', 'modulemodel.linear.Linear.0.forward_input.2.0', "<class 'int'>", "<class 'int'>", '[]', '[]', None, None, 'Pass', 'None'], ['modulemodel.linear.Linear.0.forward_input.2.1', 'modulemodel.linear.Linear.0.forward_input.2.1', "<class 'int'>", "<class 'int'>", '[]', '[]', None, None, 'Pass', 'None'], ['modulemodel.linear.Linear.0.forward_output.0', 'modulemodel.linear.Linear.0.forward_output.0', 'torch.float32', 'torch.float32', [10, 10], [10, 10], '3e8354f5', '3e8354f5', 'Pass', 'None']]
npu_op_data = {'input_args': [{'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [10, 10], 'Max': 2.8386683464050293, 'Min': -2.158618688583374, 'Mean': 0.11464785784482956, 'Norm': 10.07983684539795, 'requires_grad': False, 'md5': '7f84caad'}, None, [{'type': 'int', 'value': 2}, {'type': 'int', 'value': 2}]], 'input_kwargs': {}, 'output': [{'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [10, 10], 'Max': 1.1663073301315308, 'Min': -1.6045000553131104, 'Mean': -0.1430426388978958, 'Norm': 6.108779430389404, 'requires_grad': True, 'md5': '3e8354f5'}]}
result_1 = [{'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [10, 10], 'Max': 2.8386683464050293, 'Min': -2.158618688583374, 'Mean': 0.11464785784482956, 'Norm': 10.07983684539795, 'requires_grad': False, 'md5': '7f84caad', 'full_op_name': 'modulemodel.linear.Linear.0.forward_input.0'}, {'full_op_name': 'modulemodel.linear.Linear.0.forward_input.1', 'Max': None, 'Min': None, 'Mean': None, 'Norm': None, 'dtype': None, 'shape': None, 'md5': None, 'data_name': '-1'}, {'full_op_name': 'modulemodel.linear.Linear.0.forward_input.2.0', 'dtype': "<class 'int'>", 'shape': '[]', 'md5': None, 'Max': 2, 'Min': 2, 'Mean': 2, 'Norm': 2, 'data_name': '-1'}, {'full_op_name': 'modulemodel.linear.Linear.0.forward_input.2.1', 'dtype': "<class 'int'>", 'shape': '[]', 'md5': None, 'Max': 2, 'Min': 2, 'Mean': 2, 'Norm': 2, 'data_name': '-1'}, {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [10, 10], 'Max': 1.1663073301315308, 'Min': -1.6045000553131104, 'Mean': -0.1430426388978958, 'Norm': 6.108779430389404, 'requires_grad': True, 'md5': '3e8354f5', 'full_op_name': 'modulemodel.linear.Linear.0.forward_output.0'}]
aten_result = [['Aten__native_batch_norm_legit_functional.default_0_forward_input.0', 'Functional_batch_norm_0_forward_input.0', 'torch.float16', 'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 136.56337118148804, -124.33742618560791, -0.010397066915174946, ' ', 139.625, -127.5625, -0.0103607177734375, 3.061628818511963, -3.22507381439209, 3.634914173744619e-05, 'Warning', 'Need double check api accuracy.', 'None'], ['Aten__native_batch_norm_legit_functional.default_0_forward_input.1', 'Functional_batch_norm_0_forward_input.1', 'torch.float32', 'torch.float32', [256], [256], 2.527024927258026, -2.1782388387364335, -0.0008296193100250093, ' ', 2.5276029109954834, -2.1788690090179443, -0.0008259844034910202, 0.0005779837374575436, -0.0006301702815108001, 3.634906533989124e-06, 'Warning', 'Need double check api accuracy.', 'None'], ['Aten__native_batch_norm_legit_functional.default_0_forward_input.2', 'Functional_batch_norm_0_forward_input.2', 'torch.float32', 'torch.float32', [256], [256], 1.5384095311164856, -3.7736878395080566, -0.9390918612480164, ' ', 2.472219944000244, -2.845968723297119, -0.008756577968597412, 0.9338104128837585, 0.9277191162109375, 0.930335283279419, 'Warning', 'Need double check api accuracy.', 'None'], ['Aten__native_batch_norm_legit_functional.default_0_forward_input.3', 'Functional_batch_norm_0_forward_input.3', 'torch.float32', 'torch.float32', [256], [256], 1.763145923614502, -4.398397922515869, -1.0521326325833797, ' ', 2.763145923614502, -3.398397922515869, -0.052132632583379745, 1.0, 1.0, 1.0, 'Warning', 'Need double check api accuracy.', 'None'], ['Aten__native_batch_norm_legit_functional.default_0_forward_input.4', 'Functional_batch_norm_0_forward_input.4', 'torch.float32', 'torch.float32', [256], [256], 2.673110008239746, -3.149275064468384, 0.01613386906683445, ' ', 2.673110008239746, -3.149275064468384, 0.01613386906683445, 0.0, 0.0, 0.0, 'Warning', 'Need double check api accuracy.', 'None'], ['Aten__native_batch_norm_legit_functional.default_0_forward_output.0', 'Functional_batch_norm_0_forward_output', 'torch.float16', 'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 8.156781196594238, -4.843813419342041, -0.008758545174714527, ' ', 13.5546875, -10.640625, -0.008758544921875, 5.397906303405762, -5.796811580657959, 2.5283952709287405e-10, 'Warning', 'Need double check api accuracy.', 'None'], ['Aten__native_batch_norm_legit_functional.default_0_forward_output.1', 'Nan', 'torch.float32', 'Nan', [256], 'Nan', ' ', ' ', ' ', ' ', ' ', 0.30550330877304077, -0.24485322833061218, -0.010361209511756897, 'Nan', 'Nan', 'Nan', 'Yes', '', 'None'], ['Aten__native_batch_norm_legit_functional.default_0_forward_output.2', 'Nan', 'torch.float32', 'Nan', [256], 'Nan', ' ', ' ', ' ', ' ', ' ', 623.9192504882812, 432.96826171875, 520.2276611328125, 'Nan', 'Nan', 'Nan', 'Yes', '', 'None'], ['Aten__native_batch_norm_legit_functional.default_0_forward_output.3', 'Nan', 'torch.float32', 'Nan', [256], 'Nan', ' ', ' ', ' ', ' ', ' ', 2.4797861576080322, -3.055997371673584, -0.04795549064874649, 'Nan', 'Nan', 'Nan', 'Yes', '', 'None'], ['Aten__native_batch_norm_legit_functional.default_0_forward_output.4', 'Nan', 'torch.float32', 'Nan', [256], 'Nan', ' ', ' ', ' ', ' ', ' ', 61.7945556640625, 42.59713363647461, 52.03831481933594, 'Nan', 'Nan', 'Nan', 'Yes', '', 'None']]

npu_dict_aten = {'op_name': ['Aten__native_batch_norm_legit_functional.default_0_forward_input.0',
        'Aten__native_batch_norm_legit_functional.default_0_forward_input.1',
        'Aten__native_batch_norm_legit_functional.default_0_forward_input.2',
        'Aten__native_batch_norm_legit_functional.default_0_forward_input.3',
        'Aten__native_batch_norm_legit_functional.default_0_forward_input.4',
        'Aten__native_batch_norm_legit_functional.default_0_forward_output.0',
        'Aten__native_batch_norm_legit_functional.default_0_forward_output.1',
        'Aten__native_batch_norm_legit_functional.default_0_forward_output.2',
        'Aten__native_batch_norm_legit_functional.default_0_forward_output.3',
        'Aten__native_batch_norm_legit_functional.default_0_forward_output.4'],
        'input_struct': [('torch.float16', [256, 256, 14, 14]), ('torch.float32', [256]), ('torch.float32', [256]), ('torch.float32', [256]), ('torch.float32', [256])],
        'output_struct': [('torch.float16', [256, 256, 14, 14]), ('torch.float32', [256]), ('torch.float32', [256]), ('torch.float32', [256]), ('torch.float32', [256])],
        'summery': [[139.625, -127.5625, -0.0103607177734375],
        [2.5276029109954834, -2.1788690090179443, -0.0008259844034910202],
        [2.472219944000244, -2.845968723297119, -0.008756577968597412],
        [2.763145923614502, -3.398397922515869, -0.052132632583379745],
        [2.673110008239746, -3.149275064468384, 0.01613386906683445],
        [13.5546875, -10.640625, -0.008758544921875],
        [0.30550330877304077, -0.24485322833061218, -0.010361209511756897],
        [623.9192504882812, 432.96826171875, 520.2276611328125],
        [2.4797861576080322, -3.055997371673584, -0.04795549064874649],
        [61.7945556640625, 42.59713363647461, 52.03831481933594]]}

bench_dict_functional = {'op_name': ['Functional_batch_norm_0_forward_input.0', 'Functional_batch_norm_0_forward_input.1',
 'Functional_batch_norm_0_forward_input.2', 'Functional_batch_norm_0_forward_input.3',
  'Functional_batch_norm_0_forward_input.4', 'Functional_batch_norm_0_forward_output'],
   'input_struct': [('torch.float32', [256, 256, 14, 14]), ('torch.float32', [256]), ('torch.float32', [256]),
    ('torch.float32', [256]), ('torch.float32', [256])],
    'output_struct': [('torch.float32', [256, 256, 14, 14])],
    'summery': [[3.061628818511963, -3.22507381439209, 3.634914173744619e-05],
     [0.0005779837374575436, -0.0006301702815108001, 3.634906533989124e-06],
      [0.9338104128837585, 0.9277191162109375, 0.930335283279419],
       [1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
        [5.397906303405762, -5.796811580657959, 2.5283952709287405e-10]]
}


class TestUtilsMethods(unittest.TestCase):
    def test_correct_data(self):
        input_1 = 'NAN'
        result_1 = compare.correct_data(input_1)
        self.assertEqual(result_1, 'NAN')
        input_2 = '0.99999'
        result_2 = compare.correct_data(input_2)
        self.assertEqual(result_2, '0.99999')
        input_3 = '0.999991'
        result_3 = compare.correct_data(input_3)
        self.assertEqual(result_3, '1.0')

    def test_cosine_similarity_when_all_result_less_than_epsilon(self):
        n_value = np.array([0, 0, 0])
        b_value = np.array([0, 0, 0])
        result, message = compare.cosine_similarity(n_value, b_value)
        self.assertEqual(result, '1.0')
        self.assertEqual(message, '')

    def test_cosine_similarity_when_only_npu_result_less_than_epsilon(self):
        n_value = np.array([0, 0, 0])
        b_value = np.array([1, 2, 3])
        result, message = compare.cosine_similarity(n_value, b_value)
        self.assertEqual(result, CompareConst.NAN)
        self.assertEqual(message, 'Cannot compare by Cosine Similarity, All the data is Zero in npu dump data.')

    def test_cosine_similarity_when_only_bench_result_less_than_epsilon(self):
        n_value = np.array([1, 2, 3])
        b_value = np.array([0, 0, 0])
        result, message = compare.cosine_similarity(n_value, b_value)
        self.assertEqual(result, CompareConst.NAN)
        self.assertEqual(message, 'Cannot compare by Cosine Similarity, All the data is Zero in Bench dump data.')

    def test_cosine_similarity_when_all_result_greater_than_epsilon_with_no_nan(self):
        n_value = np.array([1, 2, 3])
        b_value = np.array([1, 2, 3])
        result, message = compare.cosine_similarity(n_value, b_value)
        
        self.assertEqual(result, '1.0')
        self.assertEqual(message, '')

    def test_cosine_similarity_when_all_result_greater_than_epsilon_with_nan(self):
        n_value = np.array([1, 2, np.nan])
        b_value = np.array([1, 2, 3])
        result, message = compare.cosine_similarity(n_value, b_value)
        self.assertEqual(result, CompareConst.NAN)
        self.assertEqual(message, 'Cannot compare by Cosine Similarity, the dump data has NaN.')

    def test_get_rmse_when_rmse_is_nan(self):
        n_value = np.array([1, 2, np.nan])
        b_value = np.array([1, 2, 3])
        rmse, message = compare.get_rmse(n_value, b_value)
        self.assertEqual(rmse, CompareConst.NAN)
        self.assertEqual(message, "")

    def test_get_mape_when_mape_is_nan(self):
        n_value = np.array([1, 2, np.nan])
        b_value = np.array([1, 2, 3])
        mape, message = compare.get_mape(n_value, b_value)
        self.assertEqual(mape, CompareConst.NAN)
        self.assertEqual(message, "")

    def test_get_max_relative_err_when_max_relative_is_nan(self):
        n_value = np.array([1, 2, np.nan])
        b_value = np.array([1, 2, 3])
        max_relative_err, message = compare.get_max_relative_err(n_value, b_value)
        self.assertEqual(max_relative_err, CompareConst.NAN)
        self.assertEqual(message, 'Cannot compare by MaxRelativeError, the data contains nan in dump data.')

    def test_get_max_relative_err_when_max_relative_is_not_nan(self):
        n_value = np.array([1, 2, 3])
        b_value = np.array([1, 2, 3])
        max_relative_err, message = compare.get_max_relative_err(n_value, b_value)
        self.assertEqual(max_relative_err, "0.000000000000")
        self.assertEqual(message, "")

    def test_check_graph_mode(self):
        op1 = "Aten"
        op2 = "torch"
        self.assertTrue(compare.check_graph_mode(op1, op2))
        self.assertTrue(compare.check_graph_mode(op2, op1))
        self.assertFalse(compare.check_graph_mode(op1, op1))
        self.assertFalse(compare.check_graph_mode(op2, op2))

    def test_check_op(self):
        fuzzy_match = False
        result = compare.check_op(npu_dict, bench_dict, fuzzy_match)
        self.assertEqual(result, True)

    def test_merge_tensor(self):
        op_dict = compare.merge_tensor(tensor_list, False, True)
        self.assertEqual(op_dict, result_op_dict)

    def test_read_op(self):
        op_name_npu = 'modulemodel.linear.Linear.0.forward'
        result = compare.read_op(npu_op_data, op_name_npu)
        self.assertEqual(result, result_1)


    def test_match_op(self):
        fuzzy_match = False
        a, b = compare.match_op([npu_dict], [bench_dict], fuzzy_match)
        self.assertEqual(a, 0)
        self.assertEqual(b, 0)

    def test_get_accuracy(self):
        result = []
        compare.get_accuracy(result, npu_dict, bench_dict, False, True)
        
        self.assertEqual(result, o_result)

    def test_get_accuracy_graph_mode(self):
        result = []
        compare.get_accuracy(result, npu_dict_aten, bench_dict_functional, True, False)
        self.assertEqual(result, aten_result)
