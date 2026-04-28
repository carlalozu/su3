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
    modules/su3v.c
)
target_include_directories(su3lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(su3lib PUBLIC OpenMP::OpenMP_C)
    target_compile_definitions(su3lib PUBLIC SU3_USE_OPENMP)
endif()


option(ENABLE_GPU_OFFLOAD "Enable OpenMP GPU Offloading" OFF)
if(ENABLE_GPU_OFFLOAD)
    target_compile_definitions(su3lib PUBLIC SU3_USE_GPU_OFFLOAD)

    add_library(su3openmp STATIC
        modules/su3v_openmp.c
    )
    target_include_directories(su3openmp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(su3openmp PUBLIC su3lib)
    set(OMP_OFFLOAD_FLAGS
        -fopenmp
        -fopenmp-targets=nvptx64-nvidia-cuda
        -Xopenmp-target=nvptx64-nvidia-cuda --offload-arch=sm_90
    )
    target_compile_options(su3lib PUBLIC ${OMP_OFFLOAD_FLAGS})
    target_link_options(su3lib PUBLIC ${OMP_OFFLOAD_FLAGS} -lmpi_gtl_cuda)
    target_compile_options(su3openmp PUBLIC ${OMP_OFFLOAD_FLAGS})
    target_link_options(su3openmp PUBLIC ${OMP_OFFLOAD_FLAGS} -lmpi_gtl_cuda)

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

    message(STATUS "CUDA native kernel enabled (SM_90)")
else()
    message(STATUS "CUDA native kernel NOT enabled")
endif()


option(ENABLE_KOKKOS "Enable Kokkos kernel" OFF)
if(ENABLE_KOKKOS)
    enable_language(CXX)

    option(Kokkos_ENABLE_CUDA   "Kokkos CUDA backend"   OFF)
    option(Kokkos_ENABLE_OPENMP "Kokkos OpenMP backend" OFF)

    if(Kokkos_ENABLE_CUDA)
        enable_language(CUDA)
    endif()

    include(FetchContent)
    FetchContent_Declare(
        Kokkos
        GIT_REPOSITORY https://github.com/kokkos/kokkos.git
        GIT_TAG        4.5.01
    )
    FetchContent_MakeAvailable(Kokkos)

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
        )
    endif()

    message(STATUS "Kokkos kernel enabled")
else()
    message(STATUS "Kokkos kernel NOT enabled")
endif()


add_subdirectory(main)
