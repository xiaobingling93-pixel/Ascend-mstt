#!/bin/bash

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"

cd ${CDIR}

gcc -fPIC -shared -o libmspti.so -I../ipc_monitor/mspti_monitor mspti.cpp
