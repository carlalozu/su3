export HOME="/scratch/calopez"
export LLVM_HOME=$HOME/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04
export PATH=$LLVM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$LLVM_HOME/lib:$LD_LIBRARY_PATH
export GCC="$(which gcc)"

repetitions=500
idx=103
avx=OFF

file_name=output/output_openmp_clang_gpu.txt
> $file_name

rm -rf build
cmake -S . -B build \
  -DCMAKE_C_COMPILER=$GCC \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=OFF -DENABLE_AVX=$avx -DENABLE_GPU_OFFLOAD=ON
cmake --build build -- -j8

echo "AVX vectorization is $avx" >> $file_name
./build/main/time_gpu $repetitions $idx >> $file_name
echo "" >> $file_name
