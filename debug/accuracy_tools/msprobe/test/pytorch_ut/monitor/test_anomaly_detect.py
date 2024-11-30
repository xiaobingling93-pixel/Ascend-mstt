import unittest
from unittest import TestCase
from unittest.mock import patch, MagicMock

from msprobe.pytorch.monitor.anomaly_detect import BaseWriterWithAD, AnomalyTurbulence, AnomalyScanner, \
                                        AnomalyDataFactory, GradAnomalyData, BaseWriterWithAD


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

    @patch("msprobe.pytorch.monitor.anomaly_detect.logger")
    def test_load_rules_with_missing_keys(self, mock_logger):
        specs = [
            {"rule_name": "AnomalyTurbulence"}
        ]
        rules = AnomalyScanner.load_rules(specs)

        self.assertEqual(len(rules), 0)
        mock_logger.warning.assert_called_once_with(f"Spec is missing required keys: {specs[0]}")

    @patch("msprobe.pytorch.monitor.anomaly_detect.logger")
    def test_load_rules_with_invalid_rule(self, mock_logger):
        specs = [
            {"rule_name": "InvalidRule", "args": {"threshold": 0.2}}
        ]
        rules = AnomalyScanner.load_rules(specs)

        self.assertEqual(len(rules), 0)

    def test_scan(self):
        AnomalyTurbulence_obj = AnomalyTurbulence(0.2)
        ad_rules = [AnomalyTurbulence_obj]
        expected = True, "AnomalyTurbulence"

        self.assertEqual(AnomalyScanner.scan(ad_rules, 1.0, 2.0), expected)


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

    def test_create(self):
        tag_name = "0:1.self_attention.core_attention_flash_0/rank0/output"
        message = "Rule AnomalyTurbulence reports anomaly signal in ('0:1.self_attention.core_attention_flash_0/rank0/output', 'min') at step 2."
        step = 2
        result = self.AnomalyDataFactory.create(tag_name, message, step)
        
        self.assertEqual(result.step, step)
        self.assertEqual(result.tag_name, tag_name)
        self.assertEqual(result.message, message)
        self.assertEqual(result.vpp_stage, 0)


class TestGradAnomalyData(TestCase):

    def setUp(self) -> None:
        tag_name = "0:1.self_attention.core_attention_flash_0/rank0/output"
        message = "Rule AnomalyTurbulence reports anomaly signal in ('0:1.self_attention.core_attention_flash_0/rank0/output', 'min') at step 2."
        group_mates = [0]
        self.GradAnomalyData = GradAnomalyData(tag_name=tag_name, message=message, group_mates=group_mates)

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


class TestBaseWriterWithAD(TestCase):

    def setUp(self) -> None:
        self.BaseWriter = BaseWriterWithAD('', None, None)

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
        self.BaseWriter.tag2scalars = {'tag':{'avg': 1.0, 'count': 1}}
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
