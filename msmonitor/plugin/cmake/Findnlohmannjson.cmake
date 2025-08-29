set(PACKAGE_VERSION 3.12.0)

set(PKG_NAME nlohmannjson)
set(DOWNLOAD_PATH "$ENV{PROJECT_ROOT_PATH}/third_party")
set(GIT_TAG "v3.12.0")
set(DIR_NAME "${DOWNLOAD_PATH}/nlohmann-json")

if (NOT ${PKG_NAME}_FOUND)

download_opensource_pkg(${PKG_NAME}
    GIT_TAG ${GIT_TAG}
    DOWNLOAD_PATH ${DOWNLOAD_PATH}
)

include_directories(${DIR_NAME}/include)
set(${PKG_NAME}_FOUND TRUE)

endif()
