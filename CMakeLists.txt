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


# Utility for baking in the current git commit into source code
set(PRE_CONFIGURE_FILE "lib/cmake-git/git.cc.in")
set(POST_CONFIGURE_FILE "${CMAKE_CURRENT_BINARY_DIR}/git.cc")
include(lib/cmake-git/git_watcher.cmake)
add_library(gitinfo STATIC ${POST_CONFIGURE_FILE})
target_include_directories(gitinfo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
add_dependencies(gitinfo check_git)

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
set(ENABLE_ALL_WARNINGS OFF)
add_definitions(-DJPH_ENABLE_ASSERTS -DJPH_DEBUG_RENDERER)
add_subdirectory(lib/JoltPhysics/Build joltPhysics)

add_subdirectory(lib/pmp-library)

# sudo apt install libcgal-dev libceres-dev
# For 5.6.1 I had to install through the debian channel, it's not available yet
add_definitions(-DCGAL_DISABLE_ROUNDING_MATH_CHECK)
find_package(CGAL REQUIRED)
#set(CGAL_BINARY_DIR ${CMAKE_BINARY_DIR}/lib/CGAL)
#set(CGAL_MODULES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/CGAL-5.6.1/cmake/modules)
#set(CGAL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/CGAL-5.6.1)
#add_subdirectory(lib/CGAL-5.6.1)

# sudo apt install freeglut3-dev
#add_subdirectory(lib/pcl)
#find_package(PCL 1.12 REQUIRED)
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
        lib/glad/include
        lib/pmp-library/src
        lib/pmp-library/external/eigen-3.4.0
        lib/cmake-git
        ../libShapeDescriptor/lib/glm
        ${CMAKE_BINARY_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        #${PCL_INCLUDE_DIRS}
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
        src/*.cpp
        src/*.cxx
        src/*.cu
        lib/glad/src/gl.c)

add_executable(shapebench ${PROJECT_SOURCES} ${PROJECT_HEADERS} ${LIBRARY_SOURCES})
target_link_libraries (shapebench ShapeDescriptor OpenMP::OpenMP_CXX Jolt glfw pmp gitinfo CGAL::CGAL ceres) # lsan ${PCL_LIBRARIES}
target_compile_options(shapebench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>) #-Xcompiler=-fsanitize=leak
target_compile_options(shapebench PRIVATE -frounding-math)
#target_compile_definitions(shapebench PRIVATE ${PCL_DEFINITIONS})

# Needed because Jolt drops a bunch of architecture specific compiler flags that NVCC doesn't understand
target_compile_options(shapebench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -forward-unknown-to-host-compiler> -std=c++20)
# Manually add options required by Jolt. Using the library's original cmake file causes trouble with the cuda compiler.
#target_compile_options(shapebench PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mavx2,-mbmi,-mpopcnt,-mlzcnt,-mf16c,-DJPH_ENABLE_ASSERTS>) # -fsanitize=leak
