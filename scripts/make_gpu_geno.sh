SCRATCH="/scratch/calopez"
export HOME=SCRATCH
export LLVM_HOME=$HOME/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04
export PATH=$LLVM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$LLVM_HOME/lib:$LD_LIBRARY_PATH
export GCC="$(which clang)"

ROOT=$SCRATCH/su3
DIR=$ROOT/scripts

cd $ROOT

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

file=volume_geno_gpu
> $file.log

perl -i -pe "s/#define CACHELINE \\d+/#define CACHELINE 128/" include/global.h
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
    -DCMAKE_C_COMPILER=$GCC \
    -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=ON -DENABLE_AVX=ON -DENABLE_GPU_OFFLOAD=ON
  cmake --build build -- -j8

  ./build/main/soa_gpu 500 100 >> $file.log

done

mv $file.log $ROOT/output/$file.log

python $DIR/parse.py < $ROOT/output/$file.log > $ROOT/output/$file.csv