#!/bin/bash
# 该脚本用于CI环境，执行系统测试用例
# Copyright (c) 2025, Huawei Technologies Co., Ltd.
# All rights reserved.

# 严格模式，任何错误都会导致脚本退出
set -e

# 测试目录
ST_DIR="test/st"

# 当前目录检查
if [[ $(basename $(pwd)) != "msmonitor" ]]; then
    if [[ -d "msmonitor" ]]; then
        echo "进入msmonitor目录"
        cd msmonitor
    else
        echo "错误: 请在msmonitor目录或其父目录下运行此脚本"
        exit 1
    fi
fi

# 设置必要的环境变量
export LD_LIBRARY_PATH=third_party/dynolog/third_party/prometheus-cpp/_build/lib:$LD_LIBRARY_PATH

echo "执行系统测试 (test/st 目录)"

# 检查系统测试目录是否存在
if [[ ! -d "$ST_DIR" ]]; then
    echo "错误: 系统测试目录 $ST_DIR 不存在"
    exit 1
fi

# 查找所有以test开头的.py文件
st_files=$(find $ST_DIR -name "test*.py")

if [[ -z "$st_files" ]]; then
    echo "错误: 没有找到测试文件"
    exit 1
fi

# 执行每个测试文件，遇到失败立即停止
for test_file in $st_files; do
    echo "==============================================="
    echo "执行测试: $test_file"

    # 直接执行Python文件
    python "$test_file"
    result=$?

    if [ $result -eq 0 ]; then
        echo "[通过] 测试成功: $test_file"
    else
        echo "[失败] 测试失败: $test_file"
        echo "==============================================="
        echo "测试执行中止: 发现失败的测试"
        exit 1
    fi
done

echo "==============================================="
echo "系统测试执行完毕"
echo "[成功] 所有测试通过"
exit 0 