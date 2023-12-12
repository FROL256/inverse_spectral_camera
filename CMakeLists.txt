# Укажите минимальную версию CMake
cmake_minimum_required(VERSION 3.8)

set(CLANG_VERSION "17" CACHE STRING "clang and LLVM version to use")
set(CMAKE_C_COMPILER   "/usr/bin/clang-${CLANG_VERSION}")   # $ export CC=/usr/bin/clang-17
set(CMAKE_CXX_COMPILER "/usr/bin/clang++-${CLANG_VERSION}") # $ export CXX=/usr/bin/clang++-17
set(ENZYME_PLUGIN_DLL  "/home/frol/PROG/enzyme/build/Enzyme/ClangEnzyme-17.so")

# Укажите имя вашего проекта
project(dcam)

# Укажите исполняемый файл и исходный файл
add_executable(dcam main.cpp)

set(ENZYME_FLAGS "-fplugin=${ENZYME_PLUGIN_DLL}")
target_compile_options(dcam PUBLIC ${ENZYME_FLAGS})
