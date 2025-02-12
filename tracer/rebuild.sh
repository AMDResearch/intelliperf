export CC=clang
export CXX=clang++
cmake -B build  -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX
cmake --build build --parallel 18 --verbose