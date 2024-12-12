import unittest
import pytest
from dataclasses import dataclass

@dataclass
class InputData:
    """模拟输入数据结构"""
    row_npu: dict
    row_gpu: dict
    compare_column: dict

class TestBasePrecisionCompare(unittest.TestCase):
    class ConcretePrecisionCompare(BasePrecisionCompare):
        """具体实现类用于测试"""
        def __init__(self, input_data):
            super().__init__(input_data)
            self.compare_algorithm = "cosine_similarity"
        
        def _get_status(self, metrics, inf_nan_consistency):
            if metrics['ratio'] >= 0.9 and inf_nan_consistency:
                metrics['compare_result'] = 'SUCCESS'
            else:
                metrics['compare_result'] = 'FAILED'
            return metrics
            
        def _compute_ratio(self):
            npu_val, gpu_val = self._get_and_convert_values('value')
            ratio = min(npu_val, gpu_val) / max(npu_val, gpu_val)
            return {'ratio': ratio}, True
    
    def setUp(self):
        """准备测试数据"""
        input_data = InputData(
            row_npu={'value': '0.95', 'other': '1.5'},
            row_gpu={'value': '1.0', 'other': '1.6'},
            compare_column={}
        )
        self.compare = self.ConcretePrecisionCompare(input_data)
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.compare.row_npu, {'value': '0.95', 'other': '1.5'})
        self.assertEqual(self.compare.row_gpu, {'value': '1.0', 'other': '1.6'})
        self.assertEqual(self.compare.compare_column, {})
        self.assertEqual(self.compare.compare_algorithm, "cosine_similarity")
    
    def test_get_and_convert_values(self):
        """测试值获取和转换"""
        npu_val, gpu_val = self.compare._get_and_convert_values('value')
        self.assertEqual(npu_val, 0.95)
        self.assertEqual(gpu_val, 1.0)
        
        npu_val, gpu_val = self.compare._get_and_convert_values('other')
        self.assertEqual(npu_val, 1.5)
        self.assertEqual(gpu_val, 1.6)
    
    def test_get_and_convert_values_none_value(self):
        """测试处理空值情况"""
        input_data = InputData(
            row_npu={'value': None},
            row_gpu={'value': '1.0'},
            compare_column={}
        )
        compare = self.ConcretePrecisionCompare(input_data)
        
        with pytest.raises(ValueError) as exc_info:
            compare._get_and_convert_values('value')
        assert "NPU value for column 'value' is None" in str(exc_info.value)
    
    def test_compare_workflow_success(self):
        """测试成功的比较流程"""
        result = self.compare.compare()
        self.assertEqual(result, 'SUCCESS')
        self.assertEqual(
            self.compare.compare_column,
            {
                'ratio': 0.95,  # min(0.95, 1.0) / max(0.95, 1.0)
                'compare_result': 'SUCCESS',
                'compare_algorithm': 'cosine_similarity'
            }
        )
    
    def test_compare_workflow_failed(self):
        """测试失败的比较流程"""
        input_data = InputData(
            row_npu={'value': '0.8'},
            row_gpu={'value': '1.0'},
            compare_column={}
        )
        compare = self.ConcretePrecisionCompare(input_data)
        
        result = compare.compare()
        self.assertEqual(result, 'FAILED')
        self.assertEqual(
            compare.compare_column,
            {
                'ratio': 0.8,  # min(0.8, 1.0) / max(0.8, 1.0)
                'compare_result': 'FAILED',
                'compare_algorithm': 'cosine_similarity'
            }
        )

if __name__ == '__main__':
    unittest.main()