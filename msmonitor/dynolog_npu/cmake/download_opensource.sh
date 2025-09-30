#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <pkg_name> <path> [ <tag> ] [ <sha256_value> ]"
    exit 1
fi

pkg_name=$1
path=$2

if [ "$#" -ge 3 ]; then
    tag=$3
fi
if [ "$#" -ge 4 ]; then
    sha256_value=$4
fi

url=$(awk -F " = " '/\['${pkg_name}'\]/{a=1}a==1&&$1~/url/{print $2;exit}' config.ini)
lib_path=$MSTT_LIB_PATH
if [ -n "$lib_path" ]; then
    url=${lib_path}$(echo $url | awk -F '/' -v OFS='/' '{print $5,$8}')
fi
if [[ ! $url = https* ]]; then
    echo "The URL of $pkg_name is illegal."
    exit 1
fi

echo "Start to download ${url}..."

if [ ! -d "$path" ]; then
    echo "The specified path does not exist: $path"
    exit 1
fi
cd ${path}

echo "Start to check ${path} permission."
# 检查下载路径是否存在
if [[ ! -d "${path}" ]]; then
    echo "Error: Download directory ${path} does not exist"
    exit 1
fi

# 检查当前用户是否有写入权限
current_user=$(whoami)
current_groups=$(groups)

# 检查下载路径的属组权限
path_info=$(ls -ld "${path}")
path_permissions=$(echo "$path_info" | awk '{print $1}')
path_group=$(echo "$path_info" | awk '{print $4}')

# 分解权限位
owner_permissions=${path_permissions:2:1}  # 所有者写权限
group_permissions=${path_permissions:5:1}  # 组写权限
other_permissions=${path_permissions:8:1}  # 其他用户写权限

# 不允许其他用户有写权限
if [[ "$other_permissions" == "w" ]]; then
    echo "Error: Other users have write permission on ${path} (security risk)"
    exit 1
fi

# 检查三种可能的写入权限情况：
# 1. 用户是目录所有者且有写权限
# 2. 用户在目录属组中且属组有写权限
# 3. 其他用户有写权限
if [[ ! -w "${path}" ]]; then
    echo "Error: Current user ${current_user} has no write permission on ${path}"
    echo "Path permissions: ${path_permissions}"
    echo "Path group: ${path_group}"
    exit 1
fi

extension=$(echo "${url}" | awk -F'[./]' '{print $NF}')
if [[ "${extension}" == "gz" || "${extension}" == "zip" ]]; then
    fullname="${path}/$(basename "${url}")"
    if [[ -e ${fullname} ]]; then
        echo "Source ${fullname} is exists, will not download again."
    else
        echo "Downloading ${url} to ${fullname}"
        curl -L "${url}" -o ${fullname}
        if [ $? -eq 0 ]; then
            echo "Download successful: ${url}"
        else
            echo "Download failed: ${url}"
            exit 1
        fi
    fi

    if [[ ! -z "${sha256_value}" ]]; then
        sha256data=$(sha256sum "${fullname}" | cut -d' ' -f1)
        if [[ "${sha256data}" != "${sha256_value}" ]]; then
            echo "Failed to verify sha256: ${url}"
            exit 1
        fi
    fi

    if [[ "${extension}" == "gz" ]]; then
        tar -zxvf ${fullname} -C ./ -n > /dev/null
    elif [[ "${extension}" == "zip" ]]; then
        unzip -n ${fullname} -d ./ > /dev/null
    fi
elif [[ "${extension}" == "git" ]]; then
    repository="$(basename ${url} .git)"
    if [[ -e ${repository} ]]; then
        echo "Source ${repository} is exists, will not clone again."
    else
        if [[ -z "${tag}" ]]; then
            git clone ${url}
        else
            git clone ${url} -b "${tag}"
        fi
        if [ $? -eq 0 ]; then
            echo "Download successful: ${url}"
        else
            echo "Download failed: ${url}"
            exit 1
        fi
    fi
else
    echo "Unknown url ${url}"
    exit 1
fi
