cmake_minimum_required(VERSION 3.16)

project(su3_alg
    VERSION 1.0
    LANGUAGES C
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(su3lib
    ${PROJECT_SOURCE_DIR}/modules/ufields.c
    ${PROJECT_SOURCE_DIR}/modules/su3.c
    ${PROJECT_SOURCE_DIR}/modules/su3v.c
)

target_include_directories(su3lib
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(su3lib PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -O
    -std=c11
)


option(ENABLE_GPU_OFFLOAD "Enable OpenMP GPU Offloading" OFF)
if(ENABLE_GPU_OFFLOAD)
    set(ENABLE_OPENMP ON)
    target_compile_definitions(su3lib PUBLIC SU3_USE_GPU_OFFLOAD)

    if(CMAKE_C_COMPILER_ID MATCHES "GNU")
        target_compile_options(su3lib PRIVATE -foffload=nvptx-none)
    elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
        target_compile_options(su3lib PRIVATE -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode -Xopenmp-target)
    endif()

    message(STATUS "GPU Offloading Enabled")
    message(STATUS "AVX disabled when using GPU")
    set(ENABLE_AVX OFF)
endif()


option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)
if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        target_link_libraries(su3lib PUBLIC OpenMP::OpenMP_C)
        target_compile_definitions(su3lib PUBLIC SU3_USE_OPENMP)
    else()
        message(WARNING "OpenMP requested but not found; building without OpenMP")
    endif()
endif()


option(ENABLE_AVX "Enable AVX vectorization" ON)
if(ENABLE_AVX)
    #  AVX-512
    target_compile_options(su3lib PRIVATE -mavx512f -mavx512dq -mavx512bw
        -mavx512vl -fopt-info-vec-missed=vec_missed.txt)
    message(STATUS "AVX enabled")
else()
    # No autovectorization
    target_compile_options(su3lib PRIVATE
        -fno-tree-vectorize
        -fno-tree-slp-vectorize
    )
    message(STATUS "AVX not enabled")
endif()


add_subdirectory(main)
