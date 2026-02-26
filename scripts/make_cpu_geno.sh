export LLVM_HOME=$SCRATCH/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04
export PATH=$LLVM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$LLVM_HOME/lib:$LD_LIBRARY_PATH
export CC="$(which clang)"

ROOT=$SCRATCH/su3
DIR=$ROOT/scripts

cd $ROOT

export GOMP_CPU_AFFINITY=0-16

file=volume_geno_cpu_float
> $file.log

perl -i -pe "s/#define CACHELINE \\d+/#define CACHELINE 8/" include/global.h
grep "#define CACHELINE" include/global.h

perl -i -pe "s/#define L1 \\d+/#define L1 8/" $ROOT/include/global.h
perl -i -pe "s/#define L2 \\d+/#define L2 8/" $ROOT/include/global.h
perl -i -pe "s/#define L3 \\d+/#define L3 8/" $ROOT/include/global.h

for i in 8 16 32 64 128
do
  base_t1=$((4 * i))
  echo $base_t1
  perl -i -pe "s/#define L0 \\d+/#define L0 $base_t1/" $ROOT/include/global.h

  rm -rf build
  cmake -S . -B build \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=ON -DENABLE_AVX=OFF
  cmake --build build -- -j8

  for t in 1 2 4 8 16
  do
    export OMP_NUM_THREADS=$((t))
    echo OMP_NUM_THREADS=$OMP_NUM_THREADS
    ./build/main/soa_cpu 500 100 >> $file.log
  done

done

mv $file.log $ROOT/output/$file.log

python $DIR/parse.py < $ROOT/output/$file.log > $ROOT/output/$file.csv
