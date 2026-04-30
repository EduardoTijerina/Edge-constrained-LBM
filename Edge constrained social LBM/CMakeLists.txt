cmake_minimum_required(VERSION 3.10)
project(mock_orchestrator CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Only look for the C library (libzmq)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZeroMQ REQUIRED IMPORTED_TARGET libzmq)

add_executable(mock_orchestrator main.cpp)

# Link the executable directly to the C library
target_link_libraries(mock_orchestrator PRIVATE PkgConfig::ZeroMQ)