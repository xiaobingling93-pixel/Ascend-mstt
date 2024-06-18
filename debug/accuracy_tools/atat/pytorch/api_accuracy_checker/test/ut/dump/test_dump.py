import unittest
from api_accuracy_checker.dump.dump import *

class TestDumpUtil(unittest.TestCase):
    def test_set_dump_switch(self):
        set_dump_switch("ON")
        self.assertEqual(DumpUtil.dump_switch, "ON")
        set_dump_switch("OFF")
        self.assertEqual(DumpUtil.dump_switch, "OFF")

    def test_get_dump_switch(self):
        DumpUtil.dump_switch = "ON"
        self.assertTrue(DumpUtil.get_dump_switch())
        DumpUtil.dump_switch = "OFF"
        self.assertFalse(DumpUtil.get_dump_switch())

    def test_incr_iter_num_maybe_exit(self):
        msCheckerConfig.target_iter = [5]
        msCheckerConfig.enable_dataloader = True

        DumpUtil.call_num = 6
        with self.assertRaises(Exception):
            DumpUtil.incr_iter_num_maybe_exit()

        DumpUtil.call_num = 4
        DumpUtil.incr_iter_num_maybe_exit()
        self.assertEqual(DumpUtil.dump_switch, "OFF")

        msCheckerConfig.enable_dataloader = False
        DumpUtil.call_num = 5
        DumpUtil.incr_iter_num_maybe_exit()
        self.assertEqual(DumpUtil.dump_switch, "ON")
