#!/bin/bash
CUR_DIR=$(dirname $(readlink -f $0))
TOP_DIR=${CUR_DIR}/..
TEST_DIR=${TOP_DIR}/"test"
SRC_DIR=${TOP_DIR}/../

clean() {
    cd ${TEST_DIR}

    if [ -e ${TEST_DIR}/"report" ]; then
      rm -r ${TEST_DIR}/"report"
      echo "remove last ut_report successfully."
    fi

}

run_ut() {
    export PYTHONPATH=${SRC_DIR}:${PYTHONPATH}
    python3 run_ut.py
}

main() {
    clean
    if [ "$1"x == "clean"x ]; then
      return 0
    fi

    cd ${TEST_DIR} && run_ut
}

main $@
