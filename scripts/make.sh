# export LLVM_HOME=$SCRATCH/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04
# export PATH=$LLVM_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$LLVM_HOME/lib:$LD_LIBRARY_PATH
# export GCC="$(which clang)"

cd ..

file=output/volume_saling_soa.log

# export OMP_NUM_THREADS=1


grep "#define L0" include/global.h

rm -rf build
cmake -S . -B build \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=ON -DENABLE_AVX=ON
cmake --build build -- -j8

# ./build/main/soa_cpu 500 100 >> $file

