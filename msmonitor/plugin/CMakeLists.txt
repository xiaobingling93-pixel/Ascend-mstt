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
    ${DYNOLOG_PATH}/third_party/glog/src
    ${DYNOLOG_PATH}/build/third_party/glog
)

file(GLOB_RECURSE IPC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/ipc_monitor/*.cpp)

set(SOURCES
    bindings.cpp
    ${IPC_SOURCES}
)

add_library(IPCMonitor MODULE ${SOURCES})

set_target_properties(IPCMonitor
    PROPERTIES
    OUTPUT_NAME IPCMonitor
    PREFIX  ""
)

target_link_libraries(IPCMonitor PRIVATE pybind11::module)
target_link_libraries(IPCMonitor PRIVATE ${DYNOLOG_PATH}/build/third_party/glog/libglog.a)

target_compile_options(IPCMonitor PRIVATE
    -fPIC
    -fstack-protector-all
    -ftrapv
)

target_link_options(IPCMonitor PRIVATE
    -Wl,-z,relro,-z,now,-z,noexecstack
    -s
)

install(TARGETS IPCMonitor
    DESTINATION ${CMAKE_INSTALL_PREFIX}/python-package
)