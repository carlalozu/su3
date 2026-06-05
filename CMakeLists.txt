cmake_minimum_required(VERSION 3.18)

project(su3_alg
    VERSION 1.0
    LANGUAGES C
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(su3lib
    modules/su3.c
    modules/su3prod.c
    modules/su3v.c
    modules/geometry.c
    modules/geogen.c
    modules/uidx.c
    modules/uflds.c
    modules/plaq_sum.c
    modules/ranlxd.c
    modules/ranlux_common.c
)
target_include_directories(su3lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(OpenMP REQUIRED)
target_link_libraries(su3lib PUBLIC OpenMP::OpenMP_C)
target_compile_definitions(su3lib PUBLIC SU3_USE_OPENMP)



option(ENABLE_GPU_OFFLOAD "Enable OpenMP GPU Offloading" OFF)
if(ENABLE_GPU_OFFLOAD)
    target_compile_definitions(su3lib PUBLIC SU3_USE_GPU_OFFLOAD)

    add_library(su3openmp STATIC modules/su3v_openmp.c)
    target_include_directories(su3openmp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(su3openmp PUBLIC su3lib)

    if (CMAKE_C_COMPILER_ID MATCHES "NVHPC")
        message(STATUS "Setting OpenMP compiler flags for NVHPC")
        set(OMP_OFFLOAD_FLAGS -mp=gpu -gpu=cc90)

        target_compile_options(su3lib PUBLIC ${OMP_OFFLOAD_FLAGS})
        target_link_options(su3lib PUBLIC ${OMP_OFFLOAD_FLAGS})
        target_compile_options(su3openmp PUBLIC ${OMP_OFFLOAD_FLAGS})
        target_link_options(su3openmp PUBLIC ${OMP_OFFLOAD_FLAGS})
    endif()

    if (CMAKE_C_COMPILER_ID MATCHES "CrayClang")
        message(STATUS "Setting OpenMP compiler flags for CrayClang")
        set(OMP_OFFLOAD_FLAGS
            -fopenmp
            -fopenmp-targets=nvptx64-nvidia-cuda
            -Xopenmp-target=nvptx64-nvidia-cuda --offload-arch=sm_90
        )
        target_compile_options(su3lib PUBLIC ${OMP_OFFLOAD_FLAGS})
        target_link_options(su3lib PUBLIC ${OMP_OFFLOAD_FLAGS} -lmpi_gtl_cuda)
        target_compile_options(su3openmp PUBLIC ${OMP_OFFLOAD_FLAGS})
        target_link_options(su3openmp PUBLIC ${OMP_OFFLOAD_FLAGS} -lmpi_gtl_cuda)
    endif()

    message(STATUS "GPU OpenMP Offloading Enabled (SM_90)")
else()
    message(STATUS "GPU OpenMP Offloading NOT Enabled")
endif()


option(ENABLE_AVX "Enable AVX vectorization" OFF)
if(ENABLE_AVX)
    message(STATUS "Vectorization enabled")
else()
    if(NOT ENABLE_GPU_OFFLOAD)
        target_compile_options(su3lib PUBLIC -fno-tree-vectorize -fno-tree-slp-vectorize)
        message(STATUS "CPU Vectorization NOT Enabled")
    endif()
endif()


option(ENABLE_CUDA "Enable native CUDA kernel" OFF)
if(ENABLE_CUDA)
    enable_language(CUDA CXX)

    add_library(su3cudakernel STATIC
        modules/su3v_cuda.cu
    )
    target_include_directories(su3cudakernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(su3cudakernel PUBLIC su3lib)
    set_target_properties(su3cudakernel PROPERTIES
        CUDA_ARCHITECTURES "90"
        CUDA_SEPARABLE_COMPILATION ON
    )

    target_link_libraries(su3cudakernel PUBLIC "-L${MPICH_PREFIX}/lib -lmpi_gtl_cuda")
    target_link_options(su3cudakernel PUBLIC 
        -Wl,-rpath,${MPICH_PREFIX}/lib
        -Wl,-rpath,/user-environment/linux-neoverse_v2/cray-gtl-8.1.32-6o2l3dln5utnr3ykymkc22xsfnwqvymp/lib
    )

    message(STATUS "CUDA native kernel enabled (SM_90)")
else()
    message(STATUS "CUDA native kernel NOT enabled")
endif()


option(ENABLE_KOKKOS "Enable Kokkos kernel" OFF)

if(ENABLE_KOKKOS)
  enable_language(CXX)

  # Find the system-installed Kokkos (provided by the module)
  find_package(Kokkos REQUIRED)

  # Query the backends that were baked into the installed Kokkos
  kokkos_check(DEVICES "Cuda"   RETURN_VALUE Kokkos_ENABLE_CUDA)
  kokkos_check(DEVICES "OpenMP" RETURN_VALUE Kokkos_ENABLE_OPENMP)

  if(Kokkos_ENABLE_CUDA)
    enable_language(CUDA)
  endif()

  add_library(su3kokkoskernel STATIC
    modules/su3v_kokkos.cpp
  )
  target_include_directories(su3kokkoskernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(su3kokkoskernel PUBLIC su3lib Kokkos::kokkos)

    if(Kokkos_ENABLE_CUDA)
    set_source_files_properties(modules/su3v_kokkos.cpp PROPERTIES LANGUAGE CUDA)
    set_target_properties(su3kokkoskernel PROPERTIES CUDA_ARCHITECTURES "90")
    target_compile_options(su3kokkoskernel PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:-ccbin ${CMAKE_CXX_COMPILER}>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>
    )

    target_link_options(su3kokkoskernel PRIVATE
        $<$<LINK_LANGUAGE:CUDA>:-Xlinker -fopenmp>
    )
    endif()

  message(STATUS "Kokkos kernel enabled (CUDA=${Kokkos_ENABLE_CUDA}, OpenMP=${Kokkos_ENABLE_OPENMP})")
else()
  message(STATUS "Kokkos kernel NOT enabled")
endif()


add_subdirectory(main)
