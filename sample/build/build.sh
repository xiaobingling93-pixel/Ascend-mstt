#依赖外部内容ASCEND_HOME_PATH

CUR_DIR=$(dirname $(readlink -f $0))
TOP_DIR=$(readlink -f $CUR_DIR/../)

# add
cd ${TOP_DIR}/normal_sample/vec_only
make
mv *.fatbin ${TOP_DIR}/build

# matmul
cd ${TOP_DIR}/normal_sample/cube_only
make
mv *.fatbin ${TOP_DIR}/build

# matmul_leakyrelu
cd ${TOP_DIR}/normal_sample/mix
make
mv *.fatbin ${TOP_DIR}/build

# illegal_read_and_write
cd ${TOP_DIR}/sanitizer_sample/memcheck/illegal_read_and_write
make
mv *.fatbin ${TOP_DIR}/build

# out_of_bound
cd ${TOP_DIR}/sanitizer_sample/memcheck/out_of_bound
make
mv *.fatbin ${TOP_DIR}/build

# illegal align sample for sanitizer
cd ${TOP_DIR}/sanitizer_sample/memcheck/illegal_align
make
mv *.fatbin ${TOP_DIR}/build

cd ${TOP_DIR}/sanitizer_sample/Racecheck
make
mv *.fatbin ${TOP_DIR}/build