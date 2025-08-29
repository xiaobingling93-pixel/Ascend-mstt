set(PACKAGE_VERSION 0.6.0)

set(PKG_NAME glog)
set(DOWNLOAD_PATH "$ENV{PROJECT_ROOT_PATH}/third_party")
set(GIT_TAG "v0.6.0")
set(DIR_NAME "${DOWNLOAD_PATH}/glog")

if (NOT ${PKG_NAME}_FOUND)

download_opensource_pkg(${PKG_NAME}
    GIT_TAG ${GIT_TAG}
    DOWNLOAD_PATH ${DOWNLOAD_PATH}
)

execute_process(
    WORKING_DIRECTORY ${DIR_NAME}
    COMMAND cmake -S . -B build -G "Unix Makefiles" -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=${DIR_NAME}/install -DCMAKE_INSTALL_LIBDIR=${DIR_NAME}/install/lib64 -DWITH_GFLAGS=OFF -DWITH_GTEST=OFF -DWITH_SYMBOLIZE=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    RESULT_VARIABLE RESULT
)
if (NOT RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to build glog. ${RESULT}")
endif()

execute_process(
    WORKING_DIRECTORY ${DIR_NAME}
    COMMAND cmake --build build --target install
    RESULT_VARIABLE RESULT
)
if (NOT RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to build glog. ${RESULT}")
endif()

file(GLOB GLOG_LIB "${DIR_NAME}/install/lib64/libglog.a")
if (NOT GLOG_LIB)
    message(FATAL_ERROR "Failed to build glog.")
endif()

set(${PKG_NAME}_LIBRARIES ${GLOG_LIB})
include_directories(${DIR_NAME}/install/include)
set(${PKG_NAME}_FOUND TRUE)

endif()
