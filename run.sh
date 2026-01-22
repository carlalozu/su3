
rm -rf build
cmake -S . -B build \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=OFF
cmake --build build -- -j8
./build/main/time 100 > output_openmp_disabled.txt

rm -rf build
cmake -S . -B build \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=ON
cmake --build build -- -j8

file_name=output_openmp.txt
> $file_name
for i in 1 2 4 8;
do
    export OMP_NUM_THREADS=$i
    echo "Running with $i threads" >> $file_name
    ./build/main/time  >> $file_name
    echo "" >> $file_name
done
# ./build/main/time 1000 > output_no_openmp.txt