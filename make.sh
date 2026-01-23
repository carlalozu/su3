rm -rf build
cmake -S . -B build \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=OFF
cmake --build build -- -j8
