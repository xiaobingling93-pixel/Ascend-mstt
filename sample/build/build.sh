#依赖外部内容ASCEND_HOME_PATH

CUR_DIR=$(dirname $(readlink -f $0))
TOP_DIR=$(readlink -f $CUR_DIR/../)

export TOP_DIR=${TOP_DIR}

# depandency
cd ${TOP_DIR}/third_party
make

cd ${TOP_DIR}/third_party/lib
cp -f libruntime.so.$(arch) libruntime.so
cp -f libruntime_camodel.so.$(arch) libruntime_camodel.so

# 
cd ${TOP_DIR}/normal_sample/vec_only
make
mv *.fatbin ${TOP_DIR}/build

cd ${TOP_DIR}/sanitizer_sample/Racecheck
make
mv *.fatbin ${TOP_DIR}/build