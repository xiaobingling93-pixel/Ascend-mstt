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
import time
import shutil
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def excute_cmd(cmd, timeout=30 * 60):
    """执行命令并捕获输出"""
    logging.info(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result


def kill_existing_dynolog_processes():
    """查找并杀死系统中已存在的dynolog进程"""
    logging.info("清理已存在的dynolog进程")
    
    try:
        current_pid = os.getpid()
        
        result = excute_cmd(["pgrep", "-f", "dynolog"])
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                # 跳过当前进程和父进程
                if int(pid) == current_pid or int(pid) == os.getppid():
                    continue
                
                logging.info(f"终止dynolog进程 (PID: {pid})")
                excute_cmd(["kill", pid])
    except Exception as e:
        logging.error(f"清理dynolog进程时出错: {str(e)}")


class TestBuildDynolog(unittest.TestCase):
    def test_build_dynolog_bin_and_plugin_whl_should_success(self):
        logging.info("构建dynolog二进制文件和插件")
        result = excute_cmd(["bash", "scripts/build.sh"])
        
        self.assertEqual(
            result.returncode,
            0,
            f"Build dynolog failed stdout: {result.stdout}, stderr: {result.stderr}",
        )

        dyno_path = "third_party/dynolog/build/bin/dyno"
        dynolog_path = "third_party/dynolog/build/bin/dynolog"

        logging.info(f"检查文件: {dyno_path}")
        self.assertTrue(os.path.exists(dyno_path), f"{dyno_path} does not exist")
        logging.info(f"检查文件: {dynolog_path}")
        self.assertTrue(os.path.exists(dynolog_path), f"{dynolog_path} does not exist")

        logging.info("构建插件wheel包")
        ori_dir = os.getcwd()
        os.chdir("plugin")
        result = excute_cmd(["bash", "build.sh"])
            
        self.assertEqual(
            result.returncode,
            0,
            f"Build msMonitor plugin whl failed stdout: {result.stdout}, stderr: {result.stderr}",
        )

        plugin_whl_path = glob.glob("dist/msmonitor_plugin-*.whl")[0]
        logging.info(f"检查wheel包: {plugin_whl_path}")
        self.assertTrue(
            os.path.exists(plugin_whl_path), f"{plugin_whl_path} does not exist"
        )
        os.chdir(ori_dir)

    def test_dynolog_command_communication(self):
        # 清理已存在的dynolog进程
        kill_existing_dynolog_processes()
        
        # 首先启动dynolog服务
        logging.info("准备启动dynolog服务")
        dynolog_process = None
        current_dir = os.getcwd()
        cert_dir = os.path.join(current_dir, "temp_certs")
        trace_log_dir = os.path.join(current_dir, "temp_trace_logs")
        
        try:
            # 创建证书目录
            os.makedirs(cert_dir, exist_ok=True)
            
            # 使用gen_tls_certs.sh脚本生成证书
            gen_certs_script = os.path.join(current_dir, "test/st/gen_tls_certs.sh")
            logging.info("生成TLS证书")
            
            # 保存当前目录
            saved_dir = os.getcwd()
            
            # 切换到证书目录
            os.chdir(cert_dir)
            
            # 执行证书生成脚本
            result = excute_cmd(["bash", gen_certs_script])
            self.assertEqual(
                result.returncode,
                0,
                f"生成TLS证书失败 stdout: {result.stdout}, stderr: {result.stderr}",
            )
            
            # 检查证书是否成功生成
            self.assertTrue(os.path.exists(os.path.join(cert_dir, "certs/server.crt")), "服务器证书未生成")
            self.assertTrue(os.path.exists(os.path.join(cert_dir, "certs/client.crt")), "客户端证书未生成")
            
            # 切回原目录
            os.chdir(saved_dir)
            
            # 使用生成的证书目录
            cert_path = "NO_CERTS"
            
            # 启动dynolog进程
            dynolog_cmd = ["third_party/dynolog/build/bin/dynolog", "--enable-ipc-monitor", "--certs-dir", cert_path]
            logging.info("启动dynolog进程")
            dynolog_process = subprocess.Popen(
                dynolog_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )

            # 等待dynolog启动
            logging.info("等待dynolog启动")
            time.sleep(5)
            
            # 验证dynolog进程是否正在运行
            poll_result = dynolog_process.poll()
            self.assertIsNone(
                poll_result, 
                "Dynolog process failed to start"
            )
            
            # 发送版本查询命令
            logging.info("测试版本查询命令")
            version_cmd = ["third_party/dynolog/build/bin/dyno", "--certs-dir", cert_path, "version"]
            version_result = excute_cmd(version_cmd)

            # 验证命令执行成功
            self.assertEqual(
                version_result.returncode,
                0,
                f"Dynolog version command failed stdout: {version_result.stdout}, stderr: {version_result.stderr}",
            )
            
            # 验证返回的版本信息不为空
            self.assertTrue(
                version_result.stdout.strip(),
                "Dynolog version command returned empty result"
            )
            
            # 发送状态查询命令
            logging.info("测试状态查询命令")
            status_cmd = ["third_party/dynolog/build/bin/dyno", "--certs-dir", cert_path, "status"]
            status_result = excute_cmd(status_cmd)

            # 验证命令执行成功
            self.assertEqual(
                status_result.returncode,
                0,
                f"Dynolog status command failed stdout: {status_result.stdout}, stderr: {status_result.stderr}",
            )
            
            # 验证返回的状态信息不为空
            self.assertTrue(
                status_result.stdout.strip(),
                "Dynolog status command returned empty result"
            )
            
            # 测试npu-monitor命令
            logging.info("测试npu-monitor启动命令")
            npumonitor_cmd = ["third_party/dynolog/build/bin/dyno", "--certs-dir",
                              cert_path, "npu-monitor", "--npu-monitor-start"]
            npumonitor_result = excute_cmd(npumonitor_cmd)

            # 验证命令执行成功
            self.assertEqual(
                npumonitor_result.returncode,
                0,
                "Dynolog npu-monitor command failed stdout: {}, stderr: {}".format(
                    npumonitor_result.stdout, npumonitor_result.stderr)
            )
            
            # 测试停止npu-monitor命令
            logging.info("测试npu-monitor停止命令")
            npumonitor_stop_cmd = ["third_party/dynolog/build/bin/dyno", "--certs-dir",
                                   cert_path, "npu-monitor", "--npu-monitor-stop"]
            npumonitor_stop_result = excute_cmd(npumonitor_stop_cmd)

            # 验证命令执行成功
            self.assertEqual(
                npumonitor_stop_result.returncode,
                0,
                "Dynolog npu-monitor stop command failed stdout: {}, stderr: {}".format(
                    npumonitor_stop_result.stdout, npumonitor_stop_result.stderr)
            )
            
            # 创建临时日志目录用于nputrace测试
            os.makedirs(trace_log_dir, exist_ok=True)
            trace_log_file = os.path.join(trace_log_dir, "test_trace.log")
            
            # 测试nputrace命令
            logging.info("测试nputrace命令")
            nputrace_cmd = [
                "third_party/dynolog/build/bin/dyno", 
                "--certs-dir", cert_path, 
                "nputrace", 
                "--start-step", "1", 
                "--iterations", "1", 
                "--activities", "NPU", 
                "--log-file", trace_log_file
            ]
            nputrace_result = excute_cmd(nputrace_cmd)

            # 验证命令执行成功
            self.assertEqual(
                nputrace_result.returncode,
                0,
                "Dynolog nputrace command failed stdout: {}, stderr: {}".format(
                    nputrace_result.stdout, nputrace_result.stderr)
            )
            
        finally:
            # 清理启动的dynolog进程
            if dynolog_process:
                logging.info("终止dynolog进程")
                dynolog_process.terminate()
                dynolog_process.wait(timeout=5)
            
            # 清理临时证书目录
            if os.path.exists(cert_dir):
                logging.info("清理临时目录")
                shutil.rmtree(cert_dir)
                
            # 清理临时日志目录
            if os.path.exists(trace_log_dir):
                shutil.rmtree(trace_log_dir)


if __name__ == "__main__":
    unittest.main()
