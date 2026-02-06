export HOME="/scratch/calopez"
export LLVM_HOME=$HOME/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04
export PATH=$LLVM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$LLVM_HOME/lib:$LD_LIBRARY_PATH
export GCC="$(which clang)"

cd ..

rm -rf build
cmake -S . -B build \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=ON -DENABLE_AVX=ON
cmake --build build -- -j8

./build/main/time 10 0