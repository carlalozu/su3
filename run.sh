repetitions=500
idx=103
avx=ON

file_name=output/d_output_openmp_avx_$avx.txt
> $file_name

rm -rf build
cmake -S . -B build \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=OFF -DENABLE_AVX=$avx
cmake --build build -- -j8

echo "AVX vectorization is $avx" >> $file_name
./build/main/time $repetitions $idx >> $file_name
echo "" >> $file_name

rm -rf build
cmake -S . -B build \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=ON -DENABLE_AVX=$avx
cmake --build build -- -j8

for i in 1 2 4 8 16;
do
    export OMP_NUM_THREADS=$i
    echo "Running with $i threads" >> $file_name
    echo "AVX vectorization is $avx" >> $file_name
    ./build/main/time $repetitions $idx >> $file_name
    echo "" >> $file_name
done
