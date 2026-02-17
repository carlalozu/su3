export LLVM_HOME=$SCRATCH/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04
export PATH=$LLVM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$LLVM_HOME/lib:$LD_LIBRARY_PATH
export CC="$(which clang)"

cd $SCRATCH/su3

export GOMP_CPU_AFFINITY=0-16

file=output/volume_geno_cpu.log
# > $file

perl -i -pe "s/#define CACHELINE \\d+/#define CACHELINE 8/" include/global.h
grep "#define CACHELINE" include/global.h

for i in 8 16 32 64
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

  for t in 1 2 4 8 16
  do
    export OMP_NUM_THREADS=$((t))
    echo OMP_NUM_THREADS=$OMP_NUM_THREADS
    ./build/main/soa_cpu 500 100 >> $file
  done

done