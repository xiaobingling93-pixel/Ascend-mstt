import unittest
import torch
import torch.nn.functional as F
from msprobe.pytorch.hook_module.wrap_functional import remove_dropout, get_functional_ops, \
    wrap_functional_ops_and_bind, HOOKFunctionalOP


class TestDropoutFunctions(unittest.TestCase):

    def setUp(self):
        self.input_tensor = torch.ones(10, 10)
        remove_dropout()

    def test_function_dropout_no_dropout(self):
        output = F.dropout(self.input_tensor, p = 0., training = True)
        self.assertTrue(torch.equal(self.input_tensor, output))

    def test_function_dropout_train_vs_eval(self):
        output_train = F.dropout(self.input_tensor, p = 0., training = True)
        output_eval = F.dropout(self.input_tensor, p = 0., training = False)
        self.assertTrue(torch.equal(output_train, output_eval))

    def test_function_dropout_invalid_probability(self):
        with self.assertRaises(ValueError):
            F.dropout(self.input_tensor, p = -0.1)
        with self.assertRaises(ValueError):
            F.dropout(self.input_tensor, p = 1.1)

    def test_function_dropout2d_no_dropout(self):
        output = F.dropout2d(self.input_tensor, p = 0., training = True)
        self.assertTrue(torch.equal(self.input_tensor, output))

    def test_function_dropout2d_train_vs_eval(self):
        output_train = F.dropout2d(self.input_tensor, p = 0., training = True)
        output_eval = F.dropout2d(self.input_tensor, p = 0., training = False)
        self.assertTrue(torch.equal(output_train, output_eval))

    def test_function_dropout2d_invalid_probability(self):
        with self.assertRaises(ValueError):
            F.dropout2d(self.input_tensor, p = -0.1)
        with self.assertRaises(ValueError):
            F.dropout2d(self.input_tensor, p = 1.1)

    def test_function_dropout3d_no_dropout(self):
        input_tensor_3d = self.input_tensor.unsqueeze(0)
        output = F.dropout3d(input_tensor_3d, p = 0., training = True)
        self.assertTrue(torch.equal(input_tensor_3d, output))
    
    def test_function_dropout3d_train_vs_eval(self):
        input_tensor_3d = self.input_tensor.unsqueeze(0)
        output_train = F.dropout3d(input_tensor_3d, p = 0., training = True)
        output_eval = F.dropout3d(input_tensor_3d, p = 0., training = False)
        self.assertTrue(torch.equal(output_train, output_eval))
    
    def test_function_dropout3d_invalid_probability(self):
        input_tensor_3d = self.input_tensor.unsqueeze(0)
        with self.assertRaises(ValueError):
            F.dropout3d(input_tensor_3d, p = -0.1)
        with self.assertRaises(ValueError):
            F.dropout3d(input_tensor_3d, p = 1.1)


class TestWrapFunctional(unittest.TestCase):

    def test_get_functional_ops(self):
        expected_ops = {'relu', 'sigmoid', 'softmax'}
        actual_ops = get_functional_ops()
        self.assertTrue(expected_ops.issubset(actual_ops))

    def test_wrap_functional_ops_and_bind(self):
        wrap_functional_ops_and_bind(None)
        self.assertTrue(hasattr(HOOKFunctionalOP, 'wrap_relu'))
