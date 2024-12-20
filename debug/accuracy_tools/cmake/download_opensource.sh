#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <url> <path> [ <sha256_value> ] [ <tag> ]"
    exit 1
fi

url=$1
path=$2

if [ "$#" -ge 3 ]; then
    sha256_value=$3
fi
if [ "$#" -ge 4 ]; then
    tag=$4
fi

echo "Start to download ${url}..."

if [ ! -d "$path" ]; then
    echo "The specified path does not exist: $path"
    exit 1
fi
cd ${path}

extension=$(echo "${url}" | awk -F'[./]' '{print $NF}')
if [[ "${extension}" == "gz" || "${extension}" == "zip" ]]; then
    fullname="${path}/$(basename "${url}")"
    if [[ -e ${fullname} ]]; then
        echo "Source ${fullname} is exists, will not download again."
    else
        curl -L "${url}" -o ${fullname} -k
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
else
    echo "Unknow url ${url}"
    exit 1
fi
