export HOME="/scratch/calopez"
export LLVM_HOME=$HOME/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04
export PATH=$LLVM_HOME/bin:$PATH
export GCC="$(which clang)"

# GCC="/usr/bin/gcc"
path="output"

cd ..
rm -rf build
cmake -S . -B build \
  -DCMAKE_C_COMPILER=$GCC \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=ON -DENABLE_AVX=ON -DENABLE_GPU_OFFLOAD=ON
cmake --build build -- -j8

# ./build/main/time 10 0
> $path/output_su3_time.txt
./build/main/time 10 0 >> $path/output_su3_time.txt