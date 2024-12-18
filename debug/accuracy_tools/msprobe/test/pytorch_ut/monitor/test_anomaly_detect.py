import unittest
from unittest import TestCase
from unittest.mock import patch

from msprobe.pytorch.monitor.anomaly_detect import AnomalyTurbulence, AnomalyScanner, \
    AnomalyDataFactory, GradAnomalyData, BaseWriterWithAD, ScanRule, WriterInput


class TestScanRule(TestCase):
    def test_apply_not_implemented(self):
        scan_rule = ScanRule()
        with self.assertRaises(Exception) as context:
            scan_rule.apply(None, None)

        self.assertEqual(str(context.exception), "abstract method apply is not implemented")


class TestAnomalyTurbulence(TestCase):

    def setUp(self) -> None:
        self.threshold = 0.2
        self.rule = AnomalyTurbulence(self.threshold)

    def test_apply_with_positive_baseline(self):
        history = [10, 12, 14]
        cur = 16
        result = self.rule.apply(history, cur)
        self.assertTrue(result)

    def test_apply_with_non_positive_baseline(self):
        history = [0, 0, 0]
        cur = -1
        result = self.rule.apply(history, cur)
        self.assertTrue(result)


class TestAnomalyScanner(TestCase):

    def test_load_rules_with_valied_spec(self):
        specs = [
            {"rule_name": "AnomalyTurbulence", "args": {"threshold": 0.2}}
        ]
        rules = AnomalyScanner.load_rules(specs)

        self.assertEqual(len(rules), 1)
        self.assertIsInstance(rules[0], AnomalyTurbulence)
        self.assertEqual(rules[0].threshold, 0.2)

        rules = AnomalyScanner.load_rules(None)
        self.assertEqual(len(rules), 0)

    @patch("msprobe.pytorch.monitor.anomaly_detect.logger")
    def test_load_rules_with_missing_keys(self, mock_logger):
        specs = [
            {"rule_name": "AnomalyTurbulence"}
        ]
        rules = AnomalyScanner.load_rules(specs)

        self.assertEqual(len(rules), 0)
        mock_logger.warning.assert_called_once_with(f"Spec is missing required keys: {specs[0]}")

    def test_load_rules_with_invalid_rule(self):
        # test invalid rule_name
        specs = [{"rule_name": "InvalidRule", "args": {"threshold": 0.2}}]
        rules = AnomalyScanner.load_rules(specs)
        self.assertEqual(len(rules), 0)

        # test invalid args
        specs = [{"rule_name": "AnomalyTurbulence", "args": "invalid args"}]
        rules = AnomalyScanner.load_rules(specs)
        self.assertEqual(len(rules), 0)

    def test_scan(self):
        ad_rules = [AnomalyTurbulence(0.2)]
        # test scan with anomaly
        expected = True, "AnomalyTurbulence"
        self.assertEqual(AnomalyScanner.scan(ad_rules, 1.0, 2.0), expected)
        # test scan with no anomaly
        expected = False, None
        self.assertEqual(AnomalyScanner.scan(ad_rules, 1.0, 1.0), expected)


class TestAnomalyDataFactory(TestCase):

    def setUp(self) -> None:
        rank = 0
        pp_stage = 0
        group_mates = [0]
        self.AnomalyDataFactory = AnomalyDataFactory(rank, pp_stage, group_mates)

    def test_set_call_id(self):
        name2callid = {'param_name': 0}
        self.AnomalyDataFactory.set_call_id(name2callid)

        self.assertEqual(self.AnomalyDataFactory.name2callid, {'param_name': 0})

    def test_create_success(self):
        tag = ('0:1.self_attention.core_attention_flash_0/rank0/output', 'min')
        message = "Rule AnomalyTurbulence reports anomaly signal in ('0:1.self_attention.core_attention_flash_0/rank0/output', 'min') at step 2."
        step = 2
        result = self.AnomalyDataFactory.create(tag, message, step)

        self.assertEqual(result.step, step)
        self.assertEqual(result.tag_name, tag[0])
        self.assertEqual(result.message, message)
        self.assertEqual(result.vpp_stage, 0)

        # test no vpp_stage
        tag = ('1.self_attention.core_attention_flash_0/rank0/output', 'min')
        result = self.AnomalyDataFactory.create(tag, message, step)
        self.assertEqual(result.vpp_stage, 0)

    def test_create_failed(self):
        error_tag = '0:1.self_attention.core_attention_flash_0/rank0/output'
        message = "Rule AnomalyTurbulence reports anomaly signal in ('0:1.self_attention.core_attention_flash_0/rank0/output', 'min') at step 2."
        step = 2
        with self.assertRaises(Exception) as context:
            self.AnomalyDataFactory.create(error_tag, message, step)
        self.assertEqual(str(context.exception), "tag must be a tuple with length 2")


class TestGradAnomalyData(TestCase):

    def setUp(self) -> None:
        tag_name = "0:1.self_attention.core_attention_flash_0/rank0/output"
        message = "Rule AnomalyTurbulence reports anomaly signal in ('0:1.self_attention.core_attention_flash_0/rank0/output', 'min') at step 2."
        group_mates = [0]
        self.GradAnomalyData = GradAnomalyData(tag_name=tag_name, message=message, group_mates=group_mates)

    def test_get_train_stage(self):
        tag_name_list = ["0:fc2_0/rank0/input", "0:fc1.weight/rank0/post_grad", "0:fc2.weight/rank0/efxp_avg_sq", ""]
        expected_train_stage_list = [0, 1, 2, -1]
        for tag_name, expected_train_stage in zip(tag_name_list, expected_train_stage_list):
            train_stage = GradAnomalyData.get_train_stage(tag_name)
            self.assertEqual(train_stage, expected_train_stage)

    def test_to_dict(self):
        expected = {
            'rank': 0,
            'step': 0,
            'micro_step': 0,
            'pp_stage': 0,
            'vpp_stage': 0,
            'call_id': 0,
            'tag_name': "0:1.self_attention.core_attention_flash_0/rank0/output",
            'message': "Rule AnomalyTurbulence reports anomaly signal in ('0:1.self_attention.core_attention_flash_0/rank0/output', 'min') at step 2.",
            'group_mates': [0]
        }

        self.assertEqual(self.GradAnomalyData.to_dict(), expected)

    def test_get_key(self):
        expected = "0:1.self_attention.core_attention_flash_0/rank0/output_step_0_call_0"

        self.assertEqual(self.GradAnomalyData.get_key(), expected)

    def test_lt_different_step(self):
        data1 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="")
        data2 = GradAnomalyData(step=2, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="")
        self.assertLess(data1, data2)
        self.assertGreater(data2, data1)

    def test_lt_same_step_different_micro_step(self):
        data1 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="")
        data2 = GradAnomalyData(step=1, micro_step=1, vpp_stage=0, pp_stage=0, call_id=0, tag_name="")
        self.assertLess(data1, data2)
        self.assertGreater(data2, data1)

    def test_lt_same_step_same_micro_step_different_vpp_stage(self):
        # same forward
        data1 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="xxx/input")
        data2 = GradAnomalyData(step=1, micro_step=0, vpp_stage=1, pp_stage=0, call_id=0, tag_name="xxx/input")
        self.assertLess(data1, data2)
        self.assertGreater(data2, data1)

        # same backward
        data1 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="xxx/post_grad")
        data2 = GradAnomalyData(step=1, micro_step=0, vpp_stage=1, pp_stage=0, call_id=0, tag_name="xxx/post_grad")
        self.assertLess(data2, data1)
        self.assertGreater(data1, data2)

        # diff train stage
        data1 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="xxx/input")
        data2 = GradAnomalyData(step=1, micro_step=0, vpp_stage=1, pp_stage=0, call_id=0, tag_name="xxx/post_grad")
        self.assertLess(data1, data2)
        self.assertGreater(data2, data1)

    def test_lt_same_step_same_micro_step_same_vpp_stage_different_pp_stage(self):
        # same forward
        data1 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="xxx/input")
        data2 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=1, call_id=0, tag_name="xxx/input")
        self.assertLess(data1, data2)
        self.assertGreater(data2, data1)

        # same backward
        data1 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="xxx/post_grad")
        data2 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=1, call_id=0, tag_name="xxx/post_grad")
        self.assertLess(data2, data1)
        self.assertGreater(data1, data2)

        # diff train stage
        data1 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="xxx/input")
        data2 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=1, call_id=0, tag_name="xxx/post_grad")
        self.assertLess(data1, data2)
        self.assertGreater(data2, data1)

    def test_lt_same_step_same_micro_step_same_vpp_stage_same_pp_stage_different_call_id(self):
        data1 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="")
        data2 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=1, tag_name="")
        self.assertLess(data1, data2)
        self.assertGreater(data2, data1)

    def test_lt_same_data(self):
        data1 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="")
        data2 = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0, tag_name="")
        self.assertGreaterEqual(data1, data2)
        self.assertLessEqual(data1, data2)

    def test_lt_not_instance(self):
        data = GradAnomalyData(step=1, micro_step=0, vpp_stage=0, pp_stage=0, call_id=0)
        not_instance = "not an instance of GradAnomalyData"
        self.assertEqual(data.__lt__(not_instance), NotImplemented)

    def test_le_same_instance(self):
        # 测试相同实例的情况
        data1 = GradAnomalyData()
        self.assertTrue(data1 <= data1)

    def test_le_different_instance(self):
        # 测试不同实例的情况
        data1 = GradAnomalyData()
        data2 = GradAnomalyData()
        self.assertTrue(data1 <= data2)

    def test_le_not_instance(self):
        # 测试非GradAnomalyData实例的情况
        data = GradAnomalyData()
        not_instance = "Not an instance of GradAnomalyData"
        self.assertEqual(data.__le__(not_instance), NotImplemented)

    def test_le_different_instance_not_equal(self):
        # 测试不同实例且不相等的情况
        data1 = GradAnomalyData()
        data2 = GradAnomalyData()
        data2.some_attribute = "some value"
        self.assertTrue(data1 <= data2)


class TestBaseWriterWithAD(TestCase):

    def setUp(self) -> None:
        self.BaseWriter = BaseWriterWithAD(WriterInput('', None, None))

    def test_get_anomalies(self):
        expected = []

        self.assertEqual(self.BaseWriter.get_anomalies(), expected)

    def test_clear_anomalies(self):
        self.BaseWriter.anomalies = ['anomaly1', 'anomaly2']
        self.BaseWriter.clear_anomalies()

        self.assertEqual(self.BaseWriter.anomalies, [])

    @patch("msprobe.pytorch.monitor.anomaly_detect.logger")
    def test_add_scalar(self, mock_logger):
        AnomalyTurbulence_obj = AnomalyTurbulence(0.2)
        self.BaseWriter.ad_rules = [AnomalyTurbulence_obj]
        self.BaseWriter.tag2scalars = {'tag': {'avg': 1.0, 'count': 1}}
        self.BaseWriter.add_scalar('tag', 2.0)

        mock_logger.info.assert_called_once()

    def test_ad(self):
        AnomalyTurbulence_obj = AnomalyTurbulence(0.2)
        self.BaseWriter.ad_rules = [AnomalyTurbulence_obj]
        expected = True, "AnomalyTurbulence"

        self.assertEqual(self.BaseWriter._ad(2.0, 1.0), expected)

    def test_update_tag2scalars(self):
        self.BaseWriter._update_tag2scalars('tag1', 1.0)
        self.assertEqual(self.BaseWriter.tag2scalars['tag1']['avg'], 1.0)
        self.assertEqual(self.BaseWriter.tag2scalars['tag1']['count'], 1)
        self.BaseWriter._update_tag2scalars('tag1', 2.0)
        self.assertEqual(self.BaseWriter.tag2scalars['tag1']['avg'], 1.5)
        self.assertEqual(self.BaseWriter.tag2scalars['tag1']['count'], 2)


if __name__ == '__main__':
    unittest.main()
