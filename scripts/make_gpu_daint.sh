#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --account=c37
#SBATCH --environment=cpe-cray-24.07
#SBATCH --exclusive

cd $SCRATCH/su3

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
file=output/volume_daint_gpu.log
> $file

perl -i -pe "s/#define CACHELINE \\d+/#define CACHELINE 128/" include/global.h
grep "#define CACHELINE" include/global.h

for i in 16 32 64 128 256
do
  NEW_VAL=$((4 * i))
  echo $NEW_VAL
  perl -i -pe "s/#define L0 \\d+/#define L0 $NEW_VAL/" include/global.h
  grep "#define L0" include/global.h

  rm -rf build
  cmake -S . -B build \
    -DCMAKE_C_COMPILER=$GCC \
    -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=ON -DENABLE_AVX=ON -DENABLE_GPU_OFFLOAD=ON
  cmake --build build -- -j8

  ./build/main/soa_gpu 500 100 >> $file

  done

# python parse.py < ../output/volume_daint_gpu_float.log > ../output/volume_daint_gpu_float.csv