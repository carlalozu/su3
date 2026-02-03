rm -rf build
cmake -S . -B build \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=ON -DENABLE_AVX=ON
cmake --build build -- -j8

# ./build/main/time 10 0
> output_su3_aosoa.txt
./build/main/su3_aosoa 10 0 >> output_su3_aosoa.txt