#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --account=c37
#SBATCH --environment=cpe-cray-24.07
#SBATCH --exclusive

cd $SCRATCH/su3

file=output/volume_saling_soa.log
> $file

perl -i -pe "s/#define CACHELINE \\d+/#define CACHELINE 8/" include/global.h
grep "#define CACHELINE" include/global.h

for i in 1 2
do
  NEW_VAL=$((4 * i))
  echo $NEW_VAL
  perl -i -pe "s/#define L0 \\d+/#define L0 $NEW_VAL/" include/global.h
  grep "#define L0" include/global.h

  rm -rf build
  cmake -S . -B build \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=ON -DENABLE_AVX=ON
  cmake --build build -- -j8

  t=2
  export OMP_NUM_THREADS=$t
  echo OMP_NUM_THREADS=$OMP_NUM_THREADS
  ./build/main/soa_cpu 500 100 >> $file

done