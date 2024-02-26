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