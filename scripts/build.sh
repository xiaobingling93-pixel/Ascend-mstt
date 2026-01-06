#!/bin/bash
# get real path of parents' dir of this file
CUR_DIR=$(dirname $(readlink -f $0))
TOP_DIR=${CUR_DIR}/..

# store product
MSTT_TEMP_DIR=${TOP_DIR}/build/mstt
rm -rf ${MSTT_TEMP_DIR}
mkdir -p ${MSTT_TEMP_DIR}

# makeself is tool for compiling run package
MAKESELF_DIR=${TOP_DIR}/opensource/makeself

# footnote for creating run package
CREATE_RUN_SCRIPT=${MAKESELF_DIR}/makeself.sh

# footnote for controling params
CONTROL_PARAM_SCRIPT=${MAKESELF_DIR}/makeself-header.sh

# store run package
OUTPUT_DIR=${TOP_DIR}/output
mkdir -p "${OUTPUT_DIR}"

RUN_SCRIPT_DIR=${TOP_DIR}/scripts/run_script
CONF_DIR=${TOP_DIR}/scripts/conf
FILTER_PARAM_SCRIPT=${CONF_DIR}/help.info
MAIN_SCRIPT=main.sh
COMMON_SCRIPT=common.sh
INSTALL_SCRIPT=install.sh
UTILS_SCRIPT=utils.sh

MSTT_RUN_NAME="mindstudio-training-tools"

PKG_LIMIT_SIZE=524288000 # 500M

function parse_script_args() {
    if [ $# -gt 2 ]; then
        echo "[ERROR] Too many arguments. Only one or two arguments are allowed."
        exit 1
    elif [ $# -eq 2 ]; then
        VERSION="$1"
        BUILD_MODE="$2"
    elif [ $# -eq 1 ]; then
        VERSION="$1"
    fi
}

# create temp dir for product
function create_temp_dir() {
    local temp_dir=${1}
    local transplt_dir=${TOP_DIR}/msfmktransplt/src/ms_fmk_transplt

    # 1. ms_fmk_transplt
    cp -r ${transplt_dir} ${temp_dir}

    # run install scripts
    copy_script ${MAIN_SCRIPT} ${temp_dir}
    copy_script ${COMMON_SCRIPT} ${temp_dir}
#    copy_script ${UTILS_SCRIPT} ${temp_dir}
}

# copy script
function copy_script() {
    local script_name=${1}
    local temp_dir=${2}

    if [ -f "${temp_dir}/${script_name}" ]; then
        rm -f "${temp_dir}/${script_name}"
    fi

    cp ${RUN_SCRIPT_DIR}/${script_name} ${temp_dir}/${script_name}
    chmod 500 "${temp_dir}/${script_name}"
}

function get_package_name() {

      CONFIG_FILE="${CUR_DIR}/conf/version.info"
    NAME=$(grep -E '^Name=' "$CONFIG_FILE" | cut -d'=' -f2)
    VERSION=$(grep -E '^Version=' "$CONFIG_FILE" | cut -d'=' -f2)
    local os_arch=$(arch)
    echo "${NAME}_${VERSION}_linux-${os_arch}.run"
}

function create_run_package() {
    local run_name=${1}
    local temp_dir=${2}
    local main_script=${MAIN_SCRIPT}
    local package_name=$(get_package_name)

    ${CREATE_RUN_SCRIPT} \
    --header ${CONTROL_PARAM_SCRIPT} \
    --help-header ${FILTER_PARAM_SCRIPT} \
    --pigz \
    --tar-quietly \
    --complevel 4 \
    --nomd5 \
    --sha256 \
    --chown \
    ${temp_dir} \
    ${OUTPUT_DIR}/${package_name} \
    ${run_name} \
    ./${main_script}
}


function check_file_exist() {
    local temp_dir=${1}

    check_package ${temp_dir}/ms_fmk_transplt ${PKG_LIMIT_SIZE}

    check_package ${temp_dir}/${MAIN_SCRIPT} ${PKG_LIMIT_SIZE}
    check_package ${temp_dir}/${COMMON_SCRIPT} ${PKG_LIMIT_SIZE}
#    check_package ${temp_dir}/${UTILS_SCRIPT} ${PKG_LIMIT_SIZE}
}

function check_package() {
    local _path="$1"
    local _limit_size=$2
    echo "check ${_path} exists"
    # 检查路径是否存在
    if [ ! -e "${_path}" ]; then
        echo "${_path} does not exist."
        exit 1
    fi

    # 检查路径是否为文件
    if [ -f "${_path}" ]; then
        local _file_size=$(stat -c%s "${_path}")
        # 检查文件大小是否超过限制
        if [ "${_file_size}" -gt "${_limit_size}" ] || [ "${_file_size}" -eq 0 ]; then
            echo "package size exceeds limit:${_limit_size}"
            exit 1
        fi
    fi
}

function main() {
	create_temp_dir ${MSTT_TEMP_DIR}
	check_file_exist ${MSTT_TEMP_DIR}
	create_run_package ${MSTT_RUN_NAME} ${MSTT_TEMP_DIR} ${main_script}
	check_package ${OUTPUT_DIR}/$(get_package_name) ${PKG_LIMIT_SIZE}
}

main
