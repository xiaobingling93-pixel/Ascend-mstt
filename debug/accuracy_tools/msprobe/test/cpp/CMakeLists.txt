project(msprobe VERSION 1.0.0 LANGUAGES CXX C)
cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(gtest MODULE REQUIRED)
find_package(mockcpp MODULE REQUIRED)
find_package(nlohmannjson MODULE REQUIRED)
find_package(cpython MODULE REQUIRED)

add_executable(msprobe_test)
target_link_libraries(msprobe_test PRIVATE ${gtest_LIBRARIES})
target_link_libraries(msprobe_test PRIVATE ${mockcpp_LIBRARIES})
target_link_libraries(msprobe_test PRIVATE _msprobe_c)

target_include_directories(msprobe_test PRIVATE $ENV{PROJECT_ROOT_PATH}/msprobe/ccsrc)
target_include_directories(msprobe_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(msprobe_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/mock)

target_compile_definitions(msprobe_test PRIVATE __RESOURCES_PATH__="${CMAKE_CURRENT_SOURCE_DIR}/../resources")

file(GLOB_RECURSE SOURCES "*.cpp")
target_sources(msprobe_test PUBLIC ${SOURCES})
