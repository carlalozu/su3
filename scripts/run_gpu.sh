export HOME="/scratch/calopez"
export LLVM_HOME=$HOME/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04
export PATH=$LLVM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$LLVM_HOME/lib:$LD_LIBRARY_PATH
export GCC="$(which clang)"

repetitions=500
idx=103
avx=OFF

export CUDA_VISIBLE_DEVICES=0

cd ..
export LIBOMPTARGET_PROFILE=profile.json

rm -rf build
cmake -S . -B build \
  -DCMAKE_C_COMPILER=$GCC \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=ON -DENABLE_AVX=$avx -DENABLE_GPU_OFFLOAD=ON
cmake --build build -- -j8

# nsys profile --stats=true 
./build/main/time_gpu $repetitions $idx

python parse.py < ../output/volume_saling_soa_gpu.log > ../output/volume_saling_soa_gpu.csv