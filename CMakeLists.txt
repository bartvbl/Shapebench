# could they at least have given these useful names..
cmake_policy(SET CMP0058 NEW)

#
# Specify minimum CMake version and project name
#
cmake_minimum_required (VERSION 3.5)

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
add_definitions(-DspinImageWidthPixels=32)
add_subdirectory(../libShapeDescriptor libShapeDescriptor)

# Makes Jolt about 8% slower but makes simulation completely deterministic no matter where the code is run
set(CROSS_PLATFORM_DETERMINISTIC ON)
set(BUILD_SHARED_LIBS OFF)
set(OVERRIDE_CXX_FLAGS OFF)
add_definitions(-DJPH_ENABLE_ASSERTS)
add_subdirectory(lib/JoltPhysics/Build joltPhysics)

#
# Set include paths
#
include_directories (
    src/
    lib/arrrgh
    lib/tinygltf
    lib/json/include
    lib/miniball/cpp/main
    lib/JoltPhysics
    lib/v-hacd/include
    ../libShapeDescriptor/lib/glm
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
target_link_libraries (shapebench ShapeDescriptor OpenMP::OpenMP_CXX Jolt) # lsan
target_compile_options(shapebench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>) #-Xcompiler=-fsanitize=leak

# Needed because Jolt drops a bunch of architecture specific compiler flags that NVCC doesn't understand
target_compile_options(shapebench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -forward-unknown-to-host-compiler>)

# Manually add options required by Jolt. Using the library's original cmake file causes trouble with the cuda compiler.
#target_compile_options(shapebench PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mavx2,-mbmi,-mpopcnt,-mlzcnt,-mf16c,-DJPH_ENABLE_ASSERTS>) # -fsanitize=leak