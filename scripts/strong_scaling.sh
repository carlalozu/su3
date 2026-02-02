export HOME="/scratch/calopez"
export LLVM_HOME=$HOME/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04
export PATH=$LLVM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$LLVM_HOME/lib:$LD_LIBRARY_PATH
export GCC="$(which clang)"

export OMP_TARGET_OFFLOAD=MANDATORY
export LIBOMPTARGET_INFO=4

repetitions=500
idx=53
avx=OFF
date=0202
cd ..

file_name=output/st_scaling_omp_clang_noavx_$date
> "$file_name.txt"

rm -rf build
cmake -S . -B build \
  -DCMAKE_C_COMPILER=$GCC \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=OFF -DENABLE_AVX=$avx -DENABLE_GPU_OFFLOAD=OFF
cmake --build build -- -j8

./build/main/time_cpu $repetitions $idx > "$file_name.txt"
echo "" >> "$file_name.txt"

rm -rf build
cmake -S . -B build \
  -DCMAKE_C_COMPILER=$GCC \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=ON -DENABLE_AVX=$avx -DENABLE_GPU_OFFLOAD=OFF
cmake --build build -- -j8

for i in 1 2 4 8 16;
do
    export OMP_NUM_THREADS=$i
    echo "Running with $i threads" >> "$file_name.txt"
    ./build/main/time_cpu $repetitions $idx >> "$file_name.txt"
    echo "" >> "$file_name.txt"
done
