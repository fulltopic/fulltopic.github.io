cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project (ch4)

set(shared_libs rt opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc python3.9)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-g -std=c++17")
set(CMAKE_BUILD_TYPE Debug)

include_directories(./)
include_directories(/home/zf/workspaces/workspace_cpp/matplotlib-cpp)
include_directories(/usr/include/python3.9)


add_executable(orig45 ./orig4_5.cpp)
target_compile_features(orig45 PUBLIC cxx_range_for)
target_link_libraries(orig45 ${shared_libs})

add_executable(gambler ./gambler.cpp)
target_compile_features(gambler PUBLIC cxx_range_for)
target_link_libraries(gambler ${shared_libs})