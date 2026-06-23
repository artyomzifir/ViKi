cmake_minimum_required(VERSION 3.16)
project(k4a_probe)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(k4a REQUIRED)

add_executable(k4a_probe k4a_probe.cpp)
target_link_libraries(k4a_probe PRIVATE k4a::k4a)
