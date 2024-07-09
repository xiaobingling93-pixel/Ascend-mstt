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
    ignore_dirs = get_ignore_dirs(cur_dir)
    cov_dir = os.path.dirname(cur_dir)
    report_dir = os.path.join(cur_dir, "report")
    final_xml_path = os.path.join(report_dir, "final.xml")
    cov_report_path = os.path.join(report_dir, "coverage.xml")

    if os.path.exists(report_dir):
        shutil.rmtree(report_dir)
    os.makedirs(report_dir)

    cmd = ["python3", "-m", "pytest", ut_path, "--junitxml=" + final_xml_path, "--cov=" + cov_dir,
           "--cov-branch", "--cov-report=xml:" + cov_report_path] + ignore_dirs
    result_ut = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while result_ut.poll() is None:
        line = result_ut.stdout.readline().strip()
        if line:
            print_info_log(str(line))

    ut_flag = False
    if result_ut.returncode == 0:
        ut_flag = True
        print_info_log("run ut successfully.")
    else:
        print_error_log("run ut failed.")

    return ut_flag


if __name__ == "__main__":
    if run_ut():
        sys.exit(0)
    else:
        sys.exit(1)
