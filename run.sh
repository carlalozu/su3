repetitions=500
idx=50
file_name=output_openmp.txt
> $file_name

rm -rf build
cmake -S . -B build \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=OFF
cmake --build build -- -j8
./build/main/time $repetitions $idx > $file_name
echo "" >> $file_name

rm -rf build
cmake -S . -B build \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=ON
cmake --build build -- -j8

for i in 1 2 4 8 16;
do
    export OMP_NUM_THREADS=$i
    echo "Running with $i threads" >> $file_name
    ./build/main/time $repetitions $idx >> $file_name
    echo "" >> $file_name
done
# ./build/main/time 1000 > output_no_openmp.txt