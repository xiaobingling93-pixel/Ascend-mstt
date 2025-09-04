set(PACKAGE_VERSION 3.50.3)

set(PKG_NAME sqlite3)
set(DOWNLOAD_PATH "$ENV{PROJECT_ROOT_PATH}/third_party")
set(DIR_NAME "${DOWNLOAD_PATH}/sqlite-amalgamation-3500300")

if (NOT ${PKG_NAME}_FOUND)

download_opensource_pkg(${PKG_NAME}
    DOWNLOAD_PATH ${DOWNLOAD_PATH}
)

file(GLOB SQLITE3_SRC "${DIR_NAME}/sqlite3.c")
if (NOT SQLITE3_SRC)
    message(FATAL_ERROR "Failed to get sqlite3 source code.")
endif()

set(${PKG_NAME}_SOURCES ${SQLITE3_SRC})
include_directories(${DIR_NAME})
set(${PKG_NAME}_FOUND TRUE)

endif()
