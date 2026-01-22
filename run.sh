export OMP_NUM_THREADS=4

rm -rf build
cmake -S . -B build \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=ON
cmake --build build -- -j8

./build/main/time 1000 > output_openmp.txt
# ./build/main/time 1000 > output_no_openmp.txt