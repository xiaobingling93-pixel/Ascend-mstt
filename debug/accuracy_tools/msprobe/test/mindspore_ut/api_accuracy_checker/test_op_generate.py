import unittest
import tempfile
import os
import json

from msprobe.core.common.const import Const
from msprobe.mindspore.api_accuracy_checker.generate_op_script.op_generator import (
    APIInfo,
    CommonConfig,
    parse_json_config,
    OperatorScriptGenerator,
    APIExtractor,
)
from msprobe.core.common.file_utils import (
    FileOpen,
    load_json,
    save_json,
    make_dir,
    change_mode,
)
from msprobe.core.common.const import FileCheckConst

class TestCommonConfigCheckConfig(unittest.TestCase):
    def setUp(self):
        # 基本有效配置
        self.tmpdir = tempfile.TemporaryDirectory()
        self.valid = {
            "dump_json_path": None,
            "api_name": "Functional.add",
            "extract_api_path": os.path.join(self.tmpdir.name, "out.json"),
            "propagation": Const.FORWARD,
            "data_mode": "random_data",
            "random_seed": 0,
            "iter_times": 1,
        }

    def tearDown(self):
        self.tmpdir.cleanup()

    def make_cfg(self, overrides):
        cfg_dict = {**self.valid, **overrides}
        return CommonConfig(cfg_dict)

    def test_invalid_api_name_too_long(self):
        long_name = "A" * 31
        with self.assertRaises(ValueError) as cm:
            self.make_cfg({"api_name": long_name})
        self.assertIn("too long", str(cm.exception))

    def test_invalid_propagation(self):
        with self.assertRaises(ValueError):
            self.make_cfg({"propagation": "INVALID"})

    def test_invalid_data_mode(self):
        with self.assertRaises(ValueError):
            self.make_cfg({"data_mode": "not_a_mode"})

    def test_random_seed_not_int(self):
        with self.assertRaises(ValueError):
            self.make_cfg({"random_seed": "zero"})

    def test_iter_times_not_int(self):
        with self.assertRaises(ValueError):
            self.make_cfg({"iter_times": "ten"})


class TestParseJsonConfig(unittest.TestCase):
    def test_empty_path_raises(self):
        with self.assertRaises(Exception) as cm:
            parse_json_config("")  # 空路径
        self.assertIn("config_input path can not be empty", str(cm.exception))

class TestAPIExtractorExtractOp(unittest.TestCase):
    def setUp(self):
        # 准备一个 dump_json_path 文件
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dump = {
            "framework": "mindspore",
            "dump_data_dir": "/data",
            "data": {
                "Functional.add.0": {
                    Const.INPUT_ARGS: [{"data_name": "a.bin"}],
                    Const.OUTPUT: [{"data_name": "b.bin"}]
                },
                "Other.mul.1": {
                    Const.INPUT_ARGS: [{"data_name": "c.bin"}]
                }
            }
        }
        self.dump_path = os.path.join(self.tmpdir.name, "dump.json")
        with open(self.dump_path, "w") as f:
            json.dump(self.dump, f)
        # 输出路径
        self.out_path = os.path.join(self.tmpdir.name, "extract.json")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_extract_op_creates_file_with_expected_keys(self):
        extractor = APIExtractor("Functional.add", self.dump_path, self.out_path)
        extractor.extract_op()
        # 文件存在
        self.assertTrue(os.path.isfile(self.out_path))
        data = load_json(self.out_path)
        # 应包含匹配 key 以及 FRAMEWORK、REAL_DATA_PATH
        self.assertIn("Functional.add.0", data)
        self.assertEqual(data.get("framework"), "mindspore")
        self.assertEqual(data.get("real_data_path"), "/data")
        # data_name 已被拼接
        arg = data["Functional.add.0"][Const.INPUT_ARGS][0]
        self.assertEqual(arg["data_name"], os.path.join("/data", "a.bin"))

class TestAPIExtractorUpdateDataNameNested(unittest.TestCase):
    def test_update_data_name_nested_list(self):
        ex = APIExtractor("Any", None, None)
        data = {"data_name": "root"}
        nested = [ [data], [{"data_name": "leaf"}] ]
        ex.update_data_name(nested, "/base")
        # 所有层级的 data_name 都被更新
        self.assertEqual(nested[0][0]["data_name"], "/base/root")
        self.assertEqual(nested[1][0]["data_name"], "/base/leaf")

class TestOperatorScriptGeneratorSegments(unittest.TestCase):
    def test_extract_segments_invalid_length(self):
        # 既不是 4 段也不是 5 段
        t, name, order = OperatorScriptGenerator.extract_detailed_api_segments("a.b.c")
        self.assertIsNone(t)
        self.assertIsNone(name)
        self.assertIsNone(order)

class TestOperatorScriptGeneratorNestedInputs(unittest.TestCase):
    def test_generate_forward_inputs_code_nested(self):
        args = [
            {"parameter_name": "x"},
            [ {"parameter_name": "y1"}, {"parameter_name": "y2"} ],
        ]
        code = OperatorScriptGenerator.generate_forward_inputs_code(args)
        self.assertIn("x", code)
        self.assertIn("y1", code)
        self.assertIn("y2", code)

    def test_generate_gradient_inputs_code_nested(self):
        args = [
            {"parameter_name": "g1"},
            [ {"parameter_name": "g2"} ]
        ]
        code = OperatorScriptGenerator.generate_gradient_inputs_code(args)
        self.assertIn("g1", code)
        self.assertIn("g2", code)


class TestAPIInfo(unittest.TestCase):
    def test_api_type_and_supported(self):
        api = APIInfo("Functional.add.0.forward", {})
        self.assertEqual(api.api_type, "Functional")
        self.assertTrue(api.is_supported_type())

    def test_from_json_forward(self):
        data = {"Functional.add.0": {"input_args": [], "input_kwargs": {}}}
        info = APIInfo.from_json(data, Const.FORWARD)
        self.assertEqual(info.api_full_name, "Functional.add.0")
        self.assertIsNone(info.backward_info)

    def test_from_json_backward(self):
        data = {
            "Functional.add.0": {"input_args": [], "input_kwargs": {}},
            "Functional.add_grad.0": {"grad_input": []},
        }
        info = APIInfo.from_json(data, Const.BACKWARD)
        self.assertEqual(info.api_full_name, "Functional.add.0")
        self.assertIsNotNone(info.backward_info)
        self.assertEqual(info.backward_info.api_full_name, "Functional.add_grad.0")

    def test_from_json_unsupported_type(self):
        data = {"Unknown.add.0": {}}
        with self.assertRaises(ValueError):
            APIInfo.from_json(data, Const.FORWARD)


class TestCommonConfig(unittest.TestCase):
    def setUp(self):
        # create a temp directory to satisfy make_dir and path checks
        self.tmpdir = tempfile.TemporaryDirectory()
        self.extract_path = os.path.join(self.tmpdir.name, "sub", "api.json")
        # build a valid config dict
        self.config = {
            "dump_json_path": None,
            "api_name": "Functional.add",
            "extract_api_path": self.extract_path,
            "propagation": Const.FORWARD,
            "data_mode": "random_data",
            "random_seed": 1,
            "iter_times": 1,
        }
        # ensure parent dir of extract_api_path exists
        os.makedirs(os.path.dirname(self.extract_path), exist_ok=True)
        # write a dummy JSON file for parse_json_config
        self.config_file = os.path.join(self.tmpdir.name, "config.json")
        with open(self.config_file, "w") as f:
            json.dump(self.config, f)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_parse_json_config(self):
        cfg = parse_json_config(self.config_file)
        self.assertIsInstance(cfg, CommonConfig)
        self.assertEqual(cfg.api_name, "Functional.add")
        self.assertEqual(cfg.propagation, Const.FORWARD)

    def test_check_user_settings_invalid_iter(self):
        cfg = CommonConfig(self.config.copy())
        cfg.iter_times = 0
        with self.assertRaises(ValueError) as ctx:
            cfg.check_user_settings()
        self.assertIn("iter_times should be range from 1", str(ctx.exception))

    def test_check_user_settings_empty_json(self):
        # create an empty JSON file to simulate empty extract_api_path
        empty = os.path.join(self.tmpdir.name, "empty.json")
        with open(empty, "w") as f:
            json.dump({}, f)
        cfg = CommonConfig({**self.config, "extract_api_path": empty})
        with self.assertRaises(ValueError) as ctx:
            cfg.check_user_settings()
        self.assertIn("json file is empty", str(ctx.exception))


class TestOperatorScriptGenerator(unittest.TestCase):
    def test_extract_detailed_api_segments_four(self):
        t, name, order = OperatorScriptGenerator.extract_detailed_api_segments(
            "Functional.mul.1.out"
        )
        self.assertEqual((t, name, order), ("Functional", "mul", "1"))

    def test_extract_detailed_api_segments_five(self):
        t, name, order = OperatorScriptGenerator.extract_detailed_api_segments(
            "Functional.prefix.mul.2.out"
        )
        self.assertEqual((t, name, order), ("Functional", "prefix.mul", "2"))

    def test_generate_forward_inputs_code(self):
        args_info = [{"parameter_name": "x"}, {"parameter_name": "y"}]
        code = OperatorScriptGenerator.generate_forward_inputs_code(args_info)
        self.assertIn("x", code)
        self.assertIn("y", code)
        self.assertIn("ComputeElement", code)

    def test_generate_kwargs_compute_element_dict_code(self):
        code = OperatorScriptGenerator.generate_kwargs_compute_element_dict_code()
        self.assertIn("kwargs_compute_element_dict", code)
        self.assertTrue(code.strip().startswith("# ---- 构造 kwargs"))

    def test_generate_gradient_inputs_code(self):
        args_back = [{"parameter_name": "grad"}]
        code = OperatorScriptGenerator.generate_gradient_inputs_code(args_back)
        self.assertIn("grad", code)
        self.assertIn("gradient_inputs", code)

    def test_get_settings_real_data(self):
        # simulate CommonConfig-like object
        common = type("C", (), {
            "propagation": Const.FORWARD,
            "random_seed": 42,
            "data_mode": "real_data",
            "iter_times": 100
        })
        gen = OperatorScriptGenerator(common, ["a"], {"k": "v"}, None)
        settings = gen.get_settings("Functional.add.0")
        self.assertEqual(settings["iter_times"], 1)
        self.assertEqual(settings["random_seed"], 42)

    def test_get_settings_random_data(self):
        common = type("C", (), {
            "propagation": Const.FORWARD,
            "random_seed": 7,
            "data_mode": "random_data",
            "iter_times": 5
        })
        gen = OperatorScriptGenerator(common, ["a"], {"k": "v"}, None)
        settings = gen.get_settings("Tensor.sub.3")
        self.assertEqual(settings["iter_times"], 5)


class TestAPIExtractor(unittest.TestCase):
    def test_update_data_name_simple(self):
        ex = APIExtractor("Functional.add", None, None)
        data = {"data_name": "foo.bin"}
        ex.update_data_name(data, "/dumpdir")
        self.assertEqual(data["data_name"], os.path.join("/dumpdir", "foo.bin"))

    def test_load_real_data_path(self):
        ex = APIExtractor("Functional.add", None, None)
        # build a minimal value dict
        val = {
            Const.INPUT_ARGS: [{"data_name": "a.txt"}],
            Const.GRAD_INPUT: [],
            Const.INPUT: [],
            Const.OUTPUT: [],
            Const.GRAD_OUTPUT: []
        }
        out = ex.load_real_data_path(val, "/mydump")
        # ensure in-place mutation happened
        self.assertEqual(val[Const.INPUT_ARGS][0]["data_name"], "/mydump/a.txt")
        self.assertIs(out, val)


if __name__ == "__main__":
    unittest.main()
