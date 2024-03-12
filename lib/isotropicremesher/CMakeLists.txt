cmake_minimum_required(VERSION 3.1...3.19)

project(isotropicremesher)

include_directories(
    src
)

add_library(isotropicremesher STATIC 
    src/axisalignedboundingboxtree.cpp
    src/isotropichalfedgemesh.cpp
    src/isotropicremesher.cpp
)