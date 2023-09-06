# could they at least have given these useful names..
cmake_policy(SET CMP0058 NEW)

#
# Specify minimum CMake version and project name
#
cmake_minimum_required (VERSION 3.0)

project (ShapeBench C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

enable_language(CUDA)

#
# CMake setup
#
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set (CMAKE_VERBOSE_MAKEFILE 0) # 1 should be used for debugging
set (CMAKE_SUPPRESS_REGENERATION TRUE) # Suppresses ZERO_CHECK

# Method specific settings
add_definitions(-DspinImageWidthPixels=64)

add_subdirectory(../libShapeDescriptor libShapeDescriptor)
#
# Set include paths
#
include_directories (
        src/
        ../libShapeDescriptor/src/
        lib/cuda-samples/Common
        lib/arrrgh
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

#
# Add files
#
file (GLOB         LIBRARY_SOURCES )
file (GLOB_RECURSE PROJECT_HEADERS src/*.hpp
        src/*.h)
file (GLOB_RECURSE PROJECT_SOURCES
        src/benchmark-core/*.cpp
        src/benchmark-core/*.cxx
        src/benchmark-core/*.cu
        src/main.cpp)

add_executable(shapebench ${PROJECT_SOURCES} ${PROJECT_HEADERS} ${LIBRARY_SOURCES})
target_link_libraries (shapebench ShapeDescriptor)