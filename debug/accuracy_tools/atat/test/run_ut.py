import os
import shutil
import subprocess
import sys

from atat.core.log import print_info_log, print_error_log


def get_ignore_dirs(cur_dir):
    ignore_dirs = []
    try:
        import torch
        import torch_npu
    except ImportError:
        print_info_log(f"Skipping the {cur_dir}/pytorch_ut directory")
        ignore_dirs.extend(["--ignore", f"{cur_dir}/pytorch_ut"])

    try:
        import mindspore
    except ImportError:
        print_info_log(f"Skipping the {cur_dir}/mindspore_ut directory")
        ignore_dirs.extend(["--ignore", f"{cur_dir}/mindspore_ut"])

    return ignore_dirs


def run_ut():
    cur_dir = os.path.realpath(os.path.dirname(__file__))
    ut_path = cur_dir
    cov_dir = os.path.dirname(cur_dir)
    report_dir = os.path.join(cur_dir, "report")
    final_xml_path = os.path.join(report_dir, "final.xml")
    cov_report_path = os.path.join(report_dir, "coverage.xml")

    if os.path.exists(report_dir):
        shutil.rmtree(report_dir)
    os.makedirs(report_dir)

    pytest_cmd = [
                     "python3", "-m", "pytest",
                     ut_path,
                     f"--junitxml={final_xml_path}",
                     f"--cov={cov_dir}",
                     "--cov-branch",
                     f"--cov-report=xml:{cov_report_path}",
                 ]

    try:
        with subprocess.Popen(
                pytest_cmd,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
        ) as proc:
            for line in proc.stdout:
                print_info_log(line.strip())

            proc.wait()

            if proc.returncode == 0:
                print_info_log("Unit tests executed successfully.")
                return True
            else:
                print_error_log("Unit tests execution failed.")
                return False
    except Exception as e:
        print_error_log(f"An error occurred during test execution: {e}")
        return False


if __name__ == "__main__":
    if run_ut():
        sys.exit(0)
    else:
        sys.exit(1)
