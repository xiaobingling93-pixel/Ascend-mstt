#!/bin/bash
function change_mode() {
    local _mode=$1
    local _path=$2
    local _type=$3

    if [ ! x"${install_for_all}" = "x" ] && [ ${install_for_all} = y ]; then
        _mode="$(expr substr ${_mode} 1 2)$(expr substr ${_mode} 2 1)"
    fi
    if [ ${_type} = "dir" ]; then
        find "${_path}" -type d -exec chmod ${_mode} {} \; 2> /dev/null
    elif [ ${_type} = "file" ]; then
        find "${_path}" -type f -exec chmod ${_mode} {} \; 2> /dev/null
    fi
}

function change_file_mode() {
    local _mode=$1
    local _path=$2
    change_mode ${_mode} "${_path}" file
}

function change_dir_mode() {
    local _mode=$1
    local _path=$2
    change_mode ${_mode} "${_path}" dir
}

function create_file() {
    local _file=$1

    if [ ! -f "${_file}" ]; then
        touch "${_file}"
        [ $? -ne 0 ] && return 1
    fi

    chown -hf "$2" "${_file}"
    [ $? -ne 0 ] && return 1
    change_file_mode "$3" "${_file}"
    [ $? -ne 0 ] && return 1
    return 0
}

function create_folder() {
    local _path=$1

    if [ -z ${_path} ]; then
        return 1
    fi

    if [ ! -d ${_path} ]; then
        mkdir -p ${_path} >/dev/null 2>&1
        [ $? -ne 0 ] && return 1
    fi

    chown -hf $2 ${_path}
    [ $? -ne 0 ] && return 1
    change_dir_mode $3 ${_path}
    [ $? -ne 0 ] && return 1
    return 0
}