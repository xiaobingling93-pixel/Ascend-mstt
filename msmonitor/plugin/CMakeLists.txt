cmake_minimum_required(VERSION 3.16)
project(IPCMonitor)

set(CMAKE_SKIP_RPATH TRUE)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/ipc_monitor
    ${CMAKE_CURRENT_SOURCE_DIR}/ipc_monitor/metric
    ${CMAKE_CURRENT_SOURCE_DIR}/ipc_monitor/mspti_monitor
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/securec/include
    ${DYNOLOG_PATH}/third_party/glog/src
    ${DYNOLOG_PATH}/build/third_party/glog
    ${DYNOLOG_PATH}/third_party/json/single_include
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
)

add_library(IPCMonitor MODULE ${SOURCES})

set_target_properties(IPCMonitor
    PROPERTIES
    OUTPUT_NAME IPCMonitor
    PREFIX  ""
)

target_link_libraries(IPCMonitor PRIVATE
    pybind11::module
    pthread
    ${CMAKE_CURRENT_SOURCE_DIR}/stub/libmspti.so
)

target_link_libraries(IPCMonitor PRIVATE ${DYNOLOG_PATH}/build/third_party/glog/libglog.a)

target_compile_options(IPCMonitor PRIVATE
    -fPIC
    -fstack-protector-all
    -ftrapv
    $<$<NOT:$<STREQUAL:${CMAKE_BUILD_TYPE},Debug>>:-O2>
)
add_compile_options(-D_FORITFY_SOURCE=2 -O2)

target_link_options(IPCMonitor PRIVATE
    -Wl,-z,relro,-z,now,-z,noexecstack
    -s
)

install(TARGETS IPCMonitor
    DESTINATION ${CMAKE_INSTALL_PREFIX}/python-package
)
