# Copyright (c) 2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import glob
import subprocess
import unittest


def excute_cmd(cmd, timeout=30 * 60):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


class TestBuildDynolog(unittest.TestCase):
    def test_build_dynolog_bin_and_plugin_whl_should_success(self):
        result = excute_cmd(["bash", "scripts/build.sh"])
        self.assertEqual(
            result.returncode,
            0,
            f"Build dynolog failed stdout: {result.stdout}, stderr: {result.stderr}",
        )

        dyno_path = "third_party/dynolog/build/bin/dyno"
        dynolog_path = "third_party/dynolog/build/bin/dynolog"

        self.assertTrue(os.path.exists(dyno_path), f"{dyno_path} does not exist")
        self.assertTrue(os.path.exists(dynolog_path), f"{dynolog_path} does not exist")

        ori_dir = os.getcwd()
        os.chdir("plugin")
        result = excute_cmd(["python3", "setup.py", "bdist_wheel"])
        self.assertEqual(
            result.returncode,
            0,
            f"Build msMonitor plugin whl failed stdout: {result.stdout}, stderr: {result.stderr}",
        )

        plugin_whl_path = glob.glob("dist/msmonitor_plugin-*.whl")[0]
        self.assertTrue(
            os.path.exists(plugin_whl_path), f"{plugin_whl_path} does not exist"
        )
        os.chdir(ori_dir)


if __name__ == "__main__":
    unittest.main()
