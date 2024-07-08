import os
import shutil
import subprocess
import sys


def run_ut():
    cur_dir = os.path.realpath(os.path.dirname(__file__))
    top_dir = os.path.realpath(os.path.dirname(cur_dir))
    ut_path = os.path.join(cur_dir, "ut/")
    src_dir = top_dir
    report_dir = os.path.join(cur_dir, "report")

    # cleanup and recreate report dir
    if os.path.exists(report_dir):
        shutil.rmtree(report_dir)
    os.makedirs(report_dir)

    # set paths for multi-platform compatibility
    junit_report = os.path.join(report_dir, "final.xml")
    coverage_report = os.path.join(report_dir, "coverage.xml")

    cmd = [
        "python3",
        "-m", "pytest",
        ut_path,
        f"--junitxml={junit_report}",
        f"--cov={src_dir}",
        "--cov-branch",
        f"--cov-report=xml:{coverage_report}"
    ]

    result_ut = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    while result_ut.poll() is None:
        line = result_ut.stdout.readline().strip()
        if line:
            print(line)

    tests_passed = result_ut.returncode == 0
    if tests_passed:
        print("run ut successfully.")
    else:
        print("run ut failed.")

    return tests_passed


if __name__ == "__main__":
    if run_ut():
        sys.exit(0)
    else:
        sys.exit(1)
