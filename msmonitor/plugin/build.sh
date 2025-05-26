#!/bin/bash

# install pybind11
pip install pybind11

# build stub
sh ./stub/build_stub.sh

# build msmonitor_plugin wheel
python3 setup.py bdist_wheel

# find .whl files in dist
files=$(find dist -type f -name "*.whl" 2>/dev/null)
count=$(echo "$files" | wc -l)
if [ "$count" -eq 1 ]; then
    echo "find .whl in dist: $files"
else
    echo "find no or multi .whl in dist"
    exit 1
fi

# pip install whl
echo "pip install ${files}"
pip install ${files}