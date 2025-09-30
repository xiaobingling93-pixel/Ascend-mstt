
function(download_opensource_pkg pkg_name)
    message("start to download ${pkg_name}...")
    set(options)
    set(oneValueArgs GIT_TAG DOWNLOAD_PATH DIR_NAME BUILD_CMD SHA256_VALUE)
    set(multiValueArgs PATCHES)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT PKG_DOWNLOAD_PATH)
        set(PKG_DOWNLOAD_PATH "${CMAKE_SOURCE_DIR}/../third_party")
    endif()
    file(MAKE_DIRECTORY ${PKG_DOWNLOAD_PATH})

    execute_process(
        WORKING_DIRECTORY $ENV{PROJECT_ROOT_PATH}/cmake
        COMMAND bash download_opensource.sh ${pkg_name} ${PKG_DOWNLOAD_PATH} ${PKG_GIT_TAG} ${PKG_SHA256_VALUE}
        RESULT_VARIABLE RESULT
    )
    if (NOT RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to download ${pkg_name}(${RESULT}).")
    endif()
    if (PKG_BUILD_CMD)
        execute_process(COMMAND bash -c "cd ${PKG_DOWNLOAD_PATH}/${DIR_NAME};${PKG_BUILD_CMD}")
    endif()
endfunction()
