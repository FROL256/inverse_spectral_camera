# Укажите минимальную версию CMake
cmake_minimum_required(VERSION 3.8)

set(CLANG_VERSION "17" CACHE STRING "clang and LLVM version to use")
set(CMAKE_C_COMPILER   "/usr/bin/clang-${CLANG_VERSION}")   # $ export CC=/usr/bin/clang-17
set(CMAKE_CXX_COMPILER "/usr/bin/clang++-${CLANG_VERSION}") # $ export CXX=/usr/bin/clang++-17
set(ENZYME_PLUGIN_DLL  "/home/frol/PROG/enzyme/build/Enzyme/ClangEnzyme-${CLANG_VERSION}.so")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/litemath
                    ${CMAKE_CURRENT_SOURCE_DIR}/external/tinyexr)

option(USE_STB_IMAGE "Use StbImage for 'png'" ON)

if(USE_STB_IMAGE)
  add_definitions(-DUSE_STB_IMAGE)
  include_directories("${CMAKE_SOURCE_DIR}/external/litemath/external")
endif()

# sanitizer options
#include(cmake/Sanitizers.cmake)
#enable_sanitizers(project_options)

#add_compile_options(-fsanitize=address)
#add_link_options(-fsanitize=address)

project(dcam)

add_library(miniz external/tinyexr/miniz.c)

# Укажите исполняемый файл и исходный файл
add_executable(dcam main.cpp 
                    checker.cpp
                    spectrum.cpp
                    imageutils.cpp 
                    external/litemath/Image2d.cpp)

set(ENZYME_FLAGS "-fplugin=${ENZYME_PLUGIN_DLL}")
target_compile_options(dcam PUBLIC ${ENZYME_FLAGS})
target_link_libraries(dcam LINK_PUBLIC miniz)
