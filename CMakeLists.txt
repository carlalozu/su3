cmake_minimum_required(VERSION 3.16)

project(su3_alg
    VERSION 1.0
    LANGUAGES C
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(su3lib
    modules/ufields.c
    modules/su3.c
    modules/su3v.c
)


target_include_directories(su3lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_options(su3lib PUBLIC
    -Wall
    -Wextra
    -Wpedantic
    -O2
    -fstrict-aliasing
    # -Rpass-analysis=loop-vectorize
    # -Rpass=loop-vectorize
    # -Rpass-missed=loop-vectorize
    -march=native
)

option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(su3lib PUBLIC OpenMP::OpenMP_C)
    target_compile_definitions(su3lib PUBLIC SU3_USE_OPENMP)
    target_link_options(su3lib PUBLIC -fopenmp)
endif()


option(ENABLE_GPU_OFFLOAD "Enable OpenMP GPU Offloading" OFF)

if(ENABLE_GPU_OFFLOAD)
    target_compile_definitions(su3lib PUBLIC SU3_USE_GPU_OFFLOAD)

    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        # GCC Specific Flags
        set(OMP_OFFLOAD_FLAGS
            -fopenmp
            -foffload=nvptx-none="-misa=sm_90" # Replace sm_70 with your GPU arch
            -fno-stack-protector
        )
    elseif(CMAKE_C_COMPILER_ID STREQUAL "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "AppleClang")
        # Clang/LLVM Specific Flags (Your original setup)
        set(OMP_OFFLOAD_FLAGS
            -fopenmp
            -fopenmp-targets=nvptx64-nvidia-cuda
            -Xopenmp-target=nvptx64-nvidia-cuda --offload-arch=sm_90
        )
    else()
        message(WARNING "GPU Offload flags not defined for ${CMAKE_C_COMPILER_ID}. Falling back to standard -fopenmp.")
        set(OMP_OFFLOAD_FLAGS -fopenmp)
    endif()

    target_compile_options(su3lib PUBLIC ${OMP_OFFLOAD_FLAGS})
    target_link_options(su3lib PUBLIC ${OMP_OFFLOAD_FLAGS})

    message(STATUS "GPU Offloading Enabled using ${CMAKE_C_COMPILER_ID}")
else()
    message(STATUS "GPU Offloading NOT Enabled")
endif()


option(ENABLE_AVX "Enable AVX vectorization" OFF)
if(ENABLE_AVX)
    message(STATUS "Vectorization enabled")
else()
    if(NOT ENABLE_GPU_OFFLOAD)
        target_compile_options(su3lib PUBLIC -fno-tree-vectorize -fno-tree-slp-vectorize)
        message(STATUS "CPU Vectorization disabled")
    endif()
endif()


add_subdirectory(main)
