# export LLVM_HOME=$SCRATCH/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04
# export PATH=$LLVM_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$LLVM_HOME/lib:$LD_LIBRARY_PATH
export CC="$(which gcc)"

cd ..

# export OMP_NUM_THREADS=1
export GOMP_DEBUG=1
export CUDA_VISIBLE_DEVICES=0
export OMP_TARGET_OFFLOAD=MANDATORY


grep "#define L0" include/global.h

rm -rf build
cmake -S . -B build \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=ON -DENABLE_AVX=ON -DENABLE_GPU_OFFLOAD=ON
cmake --build build -- -j8

./build/main/soa_gpu 500 10

