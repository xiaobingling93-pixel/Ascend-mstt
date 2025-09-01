cmake_minimum_required(VERSION 3.16)
project(IPCMonitor)

set(CMAKE_SKIP_RPATH TRUE)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(ENV{PROJECT_ROOT_PATH} "${CMAKE_SOURCE_DIR}")
include(utils)
find_package(glog MODULE REQUIRED)
find_package(nlohmannjson MODULE REQUIRED)
find_package(sqlite3 MODULE REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/ipc_monitor
    ${CMAKE_CURRENT_SOURCE_DIR}/ipc_monitor/metric
    ${CMAKE_CURRENT_SOURCE_DIR}/ipc_monitor/mspti_monitor
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/securec/include
)

file(GLOB_RECURSE IPC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ipc_monitor/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ipc_monitor/metric/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ipc_monitor/mspti_monitor/*.cpp
)

file(GLOB_RECURSE SECUREC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/third_party/securec/src/*.c)

set(SOURCES
    bindings.cpp
    ${IPC_SOURCES}
    ${SECUREC_SOURCES}
    ${sqlite3_SOURCES}
)

add_library(IPCMonitor MODULE ${SOURCES})

set_target_properties(IPCMonitor
    PROPERTIES
    OUTPUT_NAME IPCMonitor_C
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/IPCMonitor/lib64
    PREFIX  ""
)

target_link_libraries(IPCMonitor PRIVATE
    pybind11::module
    pthread
    ${glog_LIBRARIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/libmspti.so
)

target_compile_options(IPCMonitor PRIVATE
    -fPIC
    -fstack-protector-all
    -ftrapv
)

target_link_options(IPCMonitor PRIVATE
    -Wl,-z,relro,-z,now,-z,noexecstack
    -s
)

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -O0 -g")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_FORTIFY_SOURCE=2 -O2")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -D_FORTIFY_SOURCE=2 -O2")
