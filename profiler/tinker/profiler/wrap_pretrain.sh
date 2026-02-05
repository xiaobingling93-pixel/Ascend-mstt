#! /bin/bash

# 初始化一个关联数组来存储导出的环境变量
declare -A exported_vars

# 重定义 'export' 命令以捕获环境变量
export() {
    # 遍历所有传递给 export 的参数
    for var in "$@"; do
        # 检查参数是否为 VAR=VALUE 形式
        if [[ "$var" == *=* ]]; then
            var_name="${var%%=*}"
            var_value="${var#*=}"
            # 存储到关联数组
            exported_vars["$var_name"]="$var_value"
        else
            # 仅导出变量，没有赋值
            var_name="$var"
            var_value="${!var_name}"
            exported_vars["$var_name"]="$var_value"
        fi
      # 执行实际的export命令
        builtin export "$var"
    done
}

# 重定义 'python' 命令以捕获其参数并阻止执行
python() {
    echo "python_cmd_start"
    echo "python $*"
    echo "python_cmd_end"
}

# 重定义 'torchrun' 命令以捕获其参数并阻止执行
torchrun() {
    echo "torchrun_cmd_start"
    echo "torchrun $*"
    echo "torchrun_cmd_end"
}

# 执行原始脚本
source $1

# 输出捕获的环境变量
echo "export_start"
for var in "${!exported_vars[@]}"; do
    echo "export $var=${exported_vars[$var]}"
done
echo "export_end"