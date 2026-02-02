export HOME="/scratch/calopez"
export LLVM_HOME=$HOME/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04
export PATH=$LLVM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$LLVM_HOME/lib:$LD_LIBRARY_PATH
export GCC="$(which clang)"

repetitions=10
idx=103
avx=ON
gpu=OFF
date=0202

cd ..
file_name=output/time_cpu_$date
> "$file_name.txt"

export OMP_TARGET_OFFLOAD=MANDATORY
export LIBOMPTARGET_INFO=4

rm -rf build
cmake -S . -B build \
  -DCMAKE_C_COMPILER=$GCC \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=ON -DENABLE_AVX=$avx -DENABLE_GPU_OFFLOAD=$gpu
cmake --build build -- -j8

time ./build/main/time_gpu $repetitions $idx >> "$file_name.txt"

# echo "Runing plot script with $file_name"
# python scripts/plot.py $file_name.txt $file_name.pdf 