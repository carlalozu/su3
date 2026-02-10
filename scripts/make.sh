export HOME="/scratch/calopez"
export LLVM_HOME=$HOME/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04
export PATH=$LLVM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$LLVM_HOME/lib:$LD_LIBRARY_PATH
export GCC="$(which clang)"

cd ..

file=output/volume_saling_soa.log
# > $file
for t in 1 2 4 8 16
do
  export OMP_NUM_THREADS=1

  perl -i -pe "s/#define L0 \\d+/#define L0 8/" include/global.h

  for i in 1 2 4 8 16 32
  do
    NEW_VAL=$((4 * i))
    echo $NEW_VAL
    perl -i -pe "s/#define L0 \\d+/#define L0 $NEW_VAL/" include/global.h
    grep "#define L0" include/global.h

    rm -rf build
    cmake -S . -B build \
      -DCMAKE_C_COMPILER=$GCC \
      -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=ON -DENABLE_AVX=ON
    cmake --build build -- -j8

    ./build/main/soa_cpu 500 100 >> $file

    done
done