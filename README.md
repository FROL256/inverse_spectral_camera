# inverse_spectral_camera
Inverse rendering for spectral camera 

# Build

1. install llvm-17 (both dev and not dev)
 * wget https://apt.llvm.org/llvm.sh
 * chmod +x llvm.sh
 * sudo ./llvm.sh 17
 * sudo apt-get install llvm-17-dev
 * sudo touch /usr/lib/llvm-17/bin/yaml-bench 
 * sudo apt-get install libclang-17-dev 
 * sudo apt install clang-17

2. Build Enzyme 
 * Download latest release from https://github.com/EnzymeAD/Enzyme
 * cd /path/to/Enzyme/enzyme
 * mkdir build && cd build
 * cmake -G Ninja .. -DLLVM_DIR=usr/lib/llvm-17/lib/cmake/llvm DClang_DIR=/usr/lib/cmake/clang-17
 * ninja
 * you should have 'ClangEnzyme-17.so' in 'enzyme/build/Enzyme'. You have to pass this DLL to clang via '-fplugin=...' when compile the project!

3. Build this example
 * make sure you set 'ENZYME_PLUGIN_DLL' correctly to your 'ClangEnzyme-17.so'
 * use Cmake normally 
