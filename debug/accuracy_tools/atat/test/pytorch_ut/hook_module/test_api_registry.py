import unittest
from atat.pytorch.hook_module.api_registry import ApiRegistry, torch_version_above_2, is_gpu


class TestApiRegistry(unittest.TestCase):
    #100
    def test_store_ori_attr(self):
        class A():
            a1 = 1
        class B():
            a = A()
            b1 = 1
            b2 = 2
        
        api_list = ["a.a1", "b1", "b2"]
        expect_output = {"a.a1":1, "b1":1, "b2":2}
        actual_output = dict()
        ApiRegistry.store_ori_attr(B, api_list, actual_output)
        self.assertEqual(actual_output, expect_output)

    #100
    def test_set_api_attr(self):
        class A():
            a1 = 1
        class B():
            a = A().__class__
            b1 = 1
        
        attr_dict = {"a.a2":2, "b2":2, "b3":3}
        ApiRegistry.set_api_attr(B, attr_dict)

        for k, v in attr_dict.items():
            if '.' in k:
                sub_module_name, sub_op = k.rsplit('.', 1)
                sub_module = getattr(B, sub_module_name, None)
                #print(True)
                self.assertEqual(getattr(sub_module, sub_op), v)
            else:
                self.assertEqual(getattr(B, k), v)
    
    def test_api_modularity(self):

        import torch
        import torch.distributed as dist
        #import torch_npu   #门禁没有安装torch_npu
        from atat.pytorch.hook_module.api_registry import torch_without_guard_version, npu_distributed_api, is_gpu, torch_version_above_2

        

        reg = ApiRegistry()
        attr_dict = {"b2":2, "b3":3}
        reg.tensor_hook_attr = attr_dict
        reg.torch_hook_attr = attr_dict
        reg.functional_hook_attr = attr_dict
        reg.distributed_hook_attr = attr_dict
        reg.npu_distributed_hook_attr = attr_dict
        reg.aten_hook_attr = attr_dict
        reg.vf_hook_attr = attr_dict
        reg.torch_npu_hook_attr = attr_dict

        reg.api_modularity()
        self.assertEqual(torch.Tensor.b2, 2)

        self.assertEqual(torch.b2, 2)
        self.assertEqual(torch.nn.functional.b2, 2)
        self.assertEqual(dist.b2, 2)
        self.assertEqual(dist.distributed_c10d.b2, 2)
        #if not is_gpu and not torch_without_guard_version:
            #self.assertEqual(torch_npu.distributed.b2, 2)
            #self.assertEqual(torch_npu.distributed.distributed_c10d.b2, 2)
        if torch_version_above_2:
            self.assertEqual(torch.ops.aten.b2, 2)
        self.assertEqual(torch._VF.b2, 2)
        #if not is_gpu:
            #self.assertEqual(torch_npu.b2, 2)
    

    def test_api_originality(self):
        import torch
        import torch.distributed as dist
        #import torch_npu      #门禁没有安装torch_npu
        from atat.pytorch.hook_module.api_registry import torch_without_guard_version, npu_distributed_api, is_gpu, torch_version_above_2

        

        reg = ApiRegistry()
        attr_dict = {"b2":2, "b3":3}
        reg.tensor_hook_attr = attr_dict
        reg.torch_hook_attr = attr_dict
        reg.functional_hook_attr = attr_dict
        reg.distributed_hook_attr = attr_dict
        reg.npu_distributed_hook_attr = attr_dict
        reg.aten_hook_attr = attr_dict
        reg.vf_hook_attr = attr_dict
        reg.torch_npu_hook_attr = attr_dict

        reg.api_originality()
        self.assertEqual(torch.Tensor.b2, 2)

        self.assertEqual(torch.b2, 2)
        self.assertEqual(torch.nn.functional.b2, 2)
        self.assertEqual(dist.b2, 2)
        self.assertEqual(dist.distributed_c10d.b2, 2)
        #if not is_gpu and not torch_without_guard_version:
            #self.assertEqual(torch_npu.distributed.b2, 2)
            #self.assertEqual(torch_npu.distributed.distributed_c10d.b2, 2)
        if torch_version_above_2:
            self.assertEqual(torch.ops.aten.b2, 2)
        self.assertEqual(torch._VF.b2, 2)
        #if not is_gpu:
            #self.assertEqual(torch_npu.b2, 2)

    def test_initialize_hook(self):
        def hook_test():
            pass

        reg = ApiRegistry()
        reg.initialize_hook(hook_test)
        empty_list = []
        self.assertFalse(empty_list==reg.tensor_hook_attr)
        self.assertFalse(empty_list==reg.torch_hook_attr)
        self.assertFalse(empty_list==reg.functional_hook_attr)
        self.assertFalse(empty_list==reg.distributed_hook_attr)
        self.assertFalse(empty_list==reg.npu_distributed_hook_attr)
        if torch_version_above_2:
            #print(True)
            self.assertFalse(empty_list==reg.aten_hook_attr)
        if not is_gpu:
            #print(True)
            self.assertFalse(empty_list==reg.torch_npu_hook_attr)