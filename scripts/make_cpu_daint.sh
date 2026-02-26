#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --account=c37
#SBATCH --environment=cpe-cray-24.07
#SBATCH --exclusive

ROOT=$SCRATCH/su3
DIR=$ROOT/scripts

cd $ROOT

export GOMP_CPU_AFFINITY=0-71

file=volume_daint_cpu_float
> $file.log

perl -i -pe "s/#define CACHELINE \\d+/#define CACHELINE 8/" include/global.h
grep "#define CACHELINE" include/global.h

perl -i -pe "s/#define L1 \\d+/#define L1 4/" $ROOT/include/global.h
perl -i -pe "s/#define L2 \\d+/#define L2 4/" $ROOT/include/global.h
perl -i -pe "s/#define L3 \\d+/#define L3 4/" $ROOT/include/global.h

for i in 8 16 32 64 128
do
  base_t1=$((8 * 4 * i))
  echo $base_t1
  perl -i -pe "s/#define L0 \\d+/#define L0 $base_t1/" $ROOT/include/global.h

  rm -rf build
  cmake -S . -B build \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=ON -DENABLE_AVX=OFF
  cmake --build build -- -j8

  for t in 1 2 4 8 16 32 64
  do
    export OMP_NUM_THREADS=$((t))
    echo OMP_NUM_THREADS=$OMP_NUM_THREADS
    ./build/main/soa_cpu 500 100 >> $file.log
  done

done

mv $file.log $ROOT/output/$file.log

# python $DIR/parse.py < $ROOT/output/$file.log > $ROOT/output/$file.csv