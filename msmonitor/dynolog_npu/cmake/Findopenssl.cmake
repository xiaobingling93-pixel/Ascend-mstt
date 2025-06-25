set(PACKAGE_VERSION 3.0.16)

set(PKG_NAME openssl)
set(SHA256_VALUE "47ad8d3b2745717edf612fd75366faa3da4ef36b87343632de0df2433f425721")
set(GIT_TAG "openssl-3.0.16")
set(DOWNLOAD_PATH "${CMAKE_SOURCE_DIR}/third_party")
set(DIR_NAME "${DOWNLOAD_PATH}/openssl")
set(LIBDIR "lib64")

function(download_opensource_pkg pkg_name)
    message("start to download ${pkg_name}...")
    set(options)
    set(oneValueArgs SHA256 GIT_TAG DOWNLOAD_PATH DIR_NAME BUILD_CMD)
    set(multiValueArgs PATCHES)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT PKG_DOWNLOAD_PATH)
        set(PKG_DOWNLOAD_PATH "${CMAKE_SOURCE_DIR}/third_party")
    endif()
    file(MAKE_DIRECTORY ${PKG_DOWNLOAD_PATH})

    execute_process(
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/cmake
        COMMAND bash download_opensource.sh ${pkg_name} ${PKG_DOWNLOAD_PATH} ${PKG_SHA256} ${PKG_GIT_TAG}
        RESULT_VARIABLE RESULT
    )
    if (NOT RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to download ${pkg_name}(${RESULT}).")
    endif()
    if (PKG_BUILD_CMD)
        execute_process(COMMAND bash -c "cd ${PKG_DOWNLOAD_PATH}/${DIR_NAME};${PKG_BUILD_CMD}")
    endif()
endfunction()

download_opensource_pkg(${PKG_NAME}
    SHA256 ${SHA256_VALUE}
    GIT_TAG ${GIT_TAG}
    DOWNLOAD_PATH ${DOWNLOAD_PATH}
)

include_directories(${DIR_NAME}/include)
set(BUILD_DEPENDENCY_PATH "${DOWNLOAD_PATH}/openssl_build_dependency")
file(GLOB OPENSSL_LIB "${BUILD_DEPENDENCY_PATH}/${LIBDIR}/libssl.a")
file(GLOB CRYPTO_LIB "${BUILD_DEPENDENCY_PATH}/${LIBDIR}/libcrypto.a")
if (OPENSSL_LIB AND CRYPTO_LIB)
    set(${PKG_NAME}_FOUND TRUE)
    set(${PKG_NAME}_LIBRARIES "${OPENSSL_LIB};${CRYPTO_LIB}")
    return()
endif()

execute_process(
    WORKING_DIRECTORY ${DIR_NAME}
    COMMAND ./config -fPIC no-shared --prefix=${BUILD_DEPENDENCY_PATH} --libdir=${LIBDIR}
    RESULT_VARIABLE RESULT
)
if (NOT RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to build openssl. ${RESULT}")
endif()

execute_process(
    WORKING_DIRECTORY ${DIR_NAME}
    COMMAND make -j16
    RESULT_VARIABLE RESULT
)
if (NOT RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to build openssl. ${RESULT}")
endif()

execute_process(
    WORKING_DIRECTORY ${DIR_NAME}
    COMMAND make install
)

file(GLOB OPENSSL_LIB "${BUILD_DEPENDENCY_PATH}/${LIBDIR}/libssl.a")
file(GLOB CRYPTO_LIB "${BUILD_DEPENDENCY_PATH}/${LIBDIR}/libcrypto.a")
if (NOT OPENSSL_LIB OR NOT CRYPTO_LIB)
    message(FATAL_ERROR "Failed to build openssl.")
endif()

set(${PKG_NAME}_LIBRARIES "${OPENSSL_LIB};${CRYPTO_LIB}")
set(${PKG_NAME}_FOUND TRUE)
