repetitions=500
idx=103
avx=OFF
gpu=OFF
date=2801

file_name=output/time_omp_$date
> "$file_name.txt"

export OMP_TARGET_OFFLOAD=MANDATORY

cd ..
rm -rf build
cmake -S . -B build \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=OFF -DENABLE_AVX=$avx -DENABLE_GPU_OFFLOAD=$gpu
cmake --build build -- -j8

./build/main/time_gpu $repetitions $idx > "$file_name.txt"
echo "" >> "$file_name.txt"

rm -rf build
cmake -S . -B build \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=ON -DENABLE_AVX=$avx -DENABLE_GPU_OFFLOAD=$gpu
cmake --build build -- -j8

for i in 1 2 4 8 16;
do
    export OMP_NUM_THREADS=$i
    echo "Running with $i threads" >> "$file_name.txt"
    ./build/main/time_gpu $repetitions $idx >> "$file_name.txt"
    echo "" >> "$file_name.txt"
done

echo "Runing plot script with $file_name"
python scripts/plot.py $file_name.txt $file_name.pdf 