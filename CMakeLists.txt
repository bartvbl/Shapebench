# could they at least have given these useful names..
cmake_policy(SET CMP0058 NEW)

#
# Specify minimum CMake version and project name
#
cmake_minimum_required (VERSION 3.0)

project (ShapeBench C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

enable_language(CUDA)
find_package(OpenMP REQUIRED)

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
        ../libShapeDescriptor/include/
        lib/cuda-samples/Common
        lib/arrrgh
        lib/tinygltf
        lib/json/include
        lib/miniball/cpp/main
        ${CMAKE_BINARY_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

#
# Add files
#
file (GLOB         LIBRARY_SOURCES )
file (GLOB_RECURSE PROJECT_HEADERS
        src/*.hpp
        src/*.h
        src/*.cuh)
file (GLOB_RECURSE PROJECT_SOURCES
        src/benchmark-core/*.cpp
        src/benchmark-core/*.cxx
        src/benchmark-core/*.cu
        src/methods/*.cpp
        src/methods/*.cxx
        src/methods/*.cu
        src/experiment-clutter/*.cpp
        src/experiment-clutter/*.cxx
        src/experiment-clutter/*.cu
        src/main.cu)

add_executable(shapebench ${PROJECT_SOURCES} ${PROJECT_HEADERS} ${LIBRARY_SOURCES})
target_link_libraries (shapebench ShapeDescriptor OpenMP::OpenMP_CXX) # lsan
target_compile_options(shapebench PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>") #-Xcompiler=-fsanitize=leak
#target_compile_options(shapebench PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-fsanitize=leak>")