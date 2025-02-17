#!/bin/bash
set -e

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCHES_DIR="${WORK_DIR}/patches"
DYNOLOG_DIR="${WORK_DIR}/third_party/dynolog"
MODIFIED_FILES_DIR="${WORK_DIR}/dynolog_npu"

mkdir -p "${PATCHES_DIR}"

generate_patches() {
    echo "Generating patches from modified files..."
    
    # 检查修改后的文件目录是否存在
    if [ ! -d "${MODIFIED_FILES_DIR}" ]; then
        echo "ERROR: dynolog_npu directory not found"
        return 1
    fi
    
    # 清理旧的patch文件
    rm -f "${PATCHES_DIR}"/*.patch
    
    # 遍历修改后的文件目录
    find "${MODIFIED_FILES_DIR}" -type f | while read modified_file; do
        # 获取相对路径
        rel_path=$(realpath --relative-to="${MODIFIED_FILES_DIR}" "${modified_file}")
        original_file="${DYNOLOG_DIR}/${rel_path}"
        
        echo "original_file: ${original_file}"
        # 检查原始文件是否存在
        if [ ! -f "${original_file}" ]; then
            echo "WARN: Original file not found: ${original_file}"

            cp "${modified_file}" "${original_file}"
            echo "Copied ${modified_file} to ${original_file}"
            continue
        fi
        
        # 生成patch文件名（将路径中的斜杠替换为下划线）
        patch_name=$(echo "${rel_path}" | sed 's/\//_/g')
        patch_file="${PATCHES_DIR}/${patch_name}.patch"
        
        echo "Generating patch for: ${rel_path}"
        
        (
            cd "${WORK_DIR}"
            diff -u "third_party/dynolog/${rel_path}" "dynolog_npu/${rel_path}" > "${patch_file}" || true
        )
        
        # 检查patch文件大小
        if [ ! -s "${patch_file}" ]; then
            rm "${patch_file}"
            echo "No differences found for: ${rel_path}"
        else
            echo "Successfully generated patch: ${patch_file}"
        fi
    done
    
    echo "Patch generation completed"
    return 0
}

generate_patches